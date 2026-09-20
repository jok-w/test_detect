"""System-Python helper: only stdlib + JetPack GI, never imports project dependencies.

Pixels use a single shared-memory slot. The control socket carries JSON metadata;
stdout/stderr are logs, so NVIDIA library prints cannot corrupt frame data.
"""
from __future__ import annotations

import argparse
import json
import mmap
import socket
import struct
import time


def load_gst(parser_name: str):
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstApp, GstVideo
    Gst.init(None)
    for name in ("filesrc", "qtdemux", parser_name, "nvv4l2decoder", "nvvidconv", "appsink"):
        if not Gst.ElementFactory.find(name):
            raise RuntimeError(f"缺少 GStreamer 插件：{name}")
    return Gst, GstVideo


def send_message(control, message):
    data = json.dumps(message, allow_nan=False).encode("utf-8")
    control.sendall(struct.pack("!I", len(data)) + data)


def copy_bgrx(data, destination, width, height, stride, offset=0):
    """Strip any Gst row padding without requiring NumPy in system Python."""
    row_size = width * 4
    if stride < row_size or offset < 0 or len(data) < offset + (height - 1) * stride + row_size:
        raise RuntimeError("解码帧内存布局无效")
    view = memoryview(data)
    if stride == row_size:
        destination[:row_size * height] = view[offset:offset + row_size * height]
    else:
        for row in range(height):
            destination[row * row_size:(row + 1) * row_size] = view[
                offset + row * stride:offset + row * stride + row_size]


def build_pipeline(Gst, path, parser_name, prefetch):
    # Set the input property through the API; filenames never enter pipeline syntax.
    pipeline = Gst.parse_launch(
        f"filesrc name=source ! qtdemux ! {parser_name} ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        f"appsink name=frames sync=false max-buffers={prefetch} drop=false "
        "enable-last-sample=false wait-on-eos=false")
    pipeline.get_by_name("source").set_property("location", path)
    return pipeline


def run(args):
    control = socket.socket(fileno=args.control_fd)
    shared = mmap.mmap(args.memory_fd, args.width * args.height * 4)
    pipeline = None
    try:
        Gst, GstVideo = load_gst(args.parser)
        pipeline = build_pipeline(Gst, args.input, args.parser, args.prefetch)
        sink = pipeline.get_by_name("frames")
        bus = pipeline.get_bus()
        if pipeline.set_state(Gst.State.READY) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("NVDEC 管线初始化失败")
        send_message(control, {"event": "ready"})
        count, playing = 0, False
        while True:
            command = control.recv(1)
            if command in (b"", b"Q"):
                break
            if command != b"N":
                raise RuntimeError("未知解码请求")
            started = time.perf_counter()
            if not playing:
                if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("NVDEC 管线启动失败")
                state_result, _, _ = pipeline.get_state(int(args.timeout * Gst.SECOND))
                if state_result not in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.NO_PREROLL):
                    error_message = bus.pop_filtered(Gst.MessageType.ERROR)
                    if error_message is not None:
                        error, debug = error_message.parse_error()
                        raise RuntimeError(f"NVDEC 管线启动失败：{error.message}；{debug}")
                    raise RuntimeError("NVDEC 管线启动超时或失败")
                playing = True
            deadline = time.monotonic() + args.timeout
            while True:
                error_message = bus.pop_filtered(Gst.MessageType.ERROR)
                if error_message is not None:
                    error, debug = error_message.parse_error()
                    raise RuntimeError(f"NVDEC 解码失败：{error.message}；{debug}")
                sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
                if sample is not None:
                    break
                # Check errors before interpreting EOS: an errored pipeline may also be stopped.
                error_message = bus.pop_filtered(Gst.MessageType.ERROR)
                if error_message is not None:
                    error, debug = error_message.parse_error()
                    raise RuntimeError(f"NVDEC 解码失败：{error.message}；{debug}")
                if sink.is_eos():
                    send_message(control, {"event": "eos", "frames": count})
                    return 0
                if time.monotonic() >= deadline:
                    raise RuntimeError("等待 NVDEC 视频帧超时")
            pulled = time.perf_counter()
            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            if info is None or (info.width, info.height) != (args.width, args.height):
                raise RuntimeError("解码分辨率与输入元数据不一致；不支持中途变更分辨率")
            buffer = sample.get_buffer()
            pts_ns = None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
            success, mapped = buffer.map(Gst.MapFlags.READ)
            if not success:
                raise RuntimeError("无法映射解码图像")
            try:
                video_meta = GstVideo.buffer_get_video_meta(buffer)
                stride = video_meta.stride[0] if video_meta else info.stride[0]
                offset = video_meta.offset[0] if video_meta else info.offset[0]
                copy_bgrx(mapped.data, shared, args.width, args.height, stride, offset)
            finally:
                buffer.unmap(mapped)
            copied = time.perf_counter()
            # Release sample references before waiting for the next request.
            sample = buffer = mapped = None
            send_message(control, {"event": "frame", "index": count,
                                   "width": args.width, "height": args.height,
                                   "pts_ns": pts_ns,
                                   "pull_ms": (pulled - started) * 1000,
                                   "copy_ms": (copied - pulled) * 1000})
            count += 1
    except Exception as error:
        try:
            send_message(control, {"event": "error", "message": str(error)})
        except OSError:
            pass
        raise
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        shared.close()
        control.close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--parser", choices=("h264parse", "h265parse"), required=True)
    parser.add_argument("--input")
    parser.add_argument("--control-fd", type=int)
    parser.add_argument("--memory-fd", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--prefetch", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=25)
    args = parser.parse_args()
    if args.check:
        Gst, _ = load_gst(args.parser)
        print(Gst.version_string())
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
