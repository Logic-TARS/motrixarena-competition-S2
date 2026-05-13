# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_socketio import SocketIO
from PIL import Image

_NS = "/"

_last_emit_error_at: dict[str, float] = {}


def _log_emit_throttled(key: str, msg: str, interval_sec: float = 2.0) -> None:
    now = time.monotonic()
    prev = _last_emit_error_at.get(key, 0.0)
    if now - prev >= interval_sec:
        _last_emit_error_at[key] = now
        print(msg)


def _socket_json_safe(obj: Any) -> Any:
    """Ensure payloads are JSON-serializable for Engine.IO (no numpy scalars/arrays)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _socket_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_socket_json_safe(v) for v in obj]
    return str(obj)


def _queue_put_latest(q: mp.Queue, item: Any, *, max_drop: int = 16) -> bool:
    """Put latest item into bounded queue, dropping oldest items if full."""
    for _ in range(max_drop):
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
    return False


@dataclass
class WebMsgBuffer:
    reset_env: bool = False
    restart_match: bool = False
    viewer_point: list[float] | None = None
    viewer_look_at: list[float] | None = None
    camera_preset: str | None = None
    teleport_cmd: tuple[str, float, float, float | None, float | None] | None = None
    spawn_points: dict[str, list[float]] | None = None
    velocity_cmds: list[tuple[str, float, float, float]] | None = None
    referee_command: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _run_webview_process(
    template_dir: str,
    allow_keyboard_control: bool,
    port: int,
    command_q: mp.Queue,
    event_q: mp.Queue,
) -> None:
    app = Flask(__name__, template_folder=template_dir)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    latest_frame_jpeg: bytes | None = None
    latest_states: dict[str, Any] = {}
    field_meta: dict[str, Any] | None = None
    lock = threading.Lock()

    def push_cmd(payload: dict[str, Any]) -> None:
        _queue_put_latest(command_q, payload, max_drop=64)

    @app.route("/")
    def index():
        return render_template("index.html", allow_keyboard_control=bool(allow_keyboard_control))

    @app.route("/api/frame.jpg")
    def api_frame_jpg():
        nonlocal latest_frame_jpeg
        with lock:
            buf = latest_frame_jpeg
        if not buf:
            return Response(status=204)
        return Response(buf, mimetype="image/jpeg")

    @app.route("/api/states")
    def api_states():
        with lock:
            out = dict(latest_states)
        return jsonify(out)

    @socketio.on("connect")
    def on_connect():
        nonlocal field_meta
        print("[MujocoWebView] Client connected")
        if field_meta is not None:
            socketio.emit("field_meta", field_meta, namespace=_NS)

    @socketio.on("reset_env")
    def on_reset():
        push_cmd({"type": "reset_env"})

    @socketio.on("restart_match")
    def on_restart_match():
        push_cmd({"type": "restart_match"})

    @socketio.on("set_viewer_point")
    def on_view_point(data):
        if isinstance(data, dict):
            push_cmd({"type": "set_viewer_point", "point": data.get("point", [3.0, 3.0, 1.0])})

    @socketio.on("set_viewer_look_at")
    def on_view_look(data):
        if isinstance(data, dict):
            push_cmd({"type": "set_viewer_look_at", "point": data.get("point", [0.0, 0.0, 1.0])})

    @socketio.on("set_camera_preset")
    def on_camera_preset(data):
        if isinstance(data, dict):
            push_cmd({"type": "set_camera_preset", "preset": data.get("preset", "Top")})

    @socketio.on("teleport_entity")
    def on_teleport(data):
        if not isinstance(data, dict):
            return
        push_cmd(
            {
                "type": "teleport_entity",
                "name": data.get("name", ""),
                "x": float(data.get("x", 0.0)),
                "y": float(data.get("y", 0.0)),
                "z": data.get("z", None),
                "theta": data.get("theta", None),
            }
        )

    @socketio.on("set_initial_positions")
    def on_set_initial_positions(data):
        if isinstance(data, dict):
            push_cmd({"type": "set_initial_positions", "spawn_points": data})

    @socketio.on("set_robot_velocity")
    def on_set_robot_velocity(data):
        if not isinstance(data, dict):
            return
        push_cmd(
            {
                "type": "set_robot_velocity",
                "name": str(data.get("name", "")),
                "vx": float(data.get("vx", 0.0)),
                "vy": float(data.get("vy", 0.0)),
                "wz": float(data.get("wz", 0.0)),
            }
        )

    @socketio.on("referee_command")
    def on_referee_command(data):
        if not isinstance(data, dict):
            return
        cmd = str(data.get("command", "")).strip()
        if cmd in ("ready", "set", "play", "finish", "stoptimer"):
            push_cmd({"type": "referee_command", "command": cmd})

    def event_pump() -> None:
        nonlocal latest_frame_jpeg, latest_states, field_meta
        while True:
            try:
                event = event_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if event is None:
                break
            if not isinstance(event, dict):
                continue
            et = event.get("type", "")
            if et == "shutdown":
                break
            if et == "frame":
                jpeg = event.get("jpeg", None)
                if isinstance(jpeg, (bytes, bytearray)):
                    jb = bytes(jpeg)
                    with lock:
                        latest_frame_jpeg = jb
                    payload = base64.b64encode(jb).decode("utf-8")
                    socketio.emit("new_frame", {"image": payload}, namespace=_NS)
            elif et == "states":
                states = event.get("states", {})
                if isinstance(states, dict):
                    with lock:
                        latest_states = dict(states)
                    socketio.emit("robot_states", states, namespace=_NS)
            elif et == "field_meta":
                fm = event.get("field_meta", None)
                if isinstance(fm, dict):
                    with lock:
                        field_meta = dict(fm)
                    socketio.emit("field_meta", field_meta, namespace=_NS)

    t = threading.Thread(target=event_pump, daemon=True)
    t.start()
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(port),
        use_reloader=False,
        debug=False,
        allow_unsafe_werkzeug=True,
    )


class MujocoLabWebView:
    def __init__(self, template_dir: Path, allow_keyboard_control: bool = False):
        self.template_dir = Path(template_dir)
        self.msg = WebMsgBuffer()
        self.allow_keyboard_control = bool(allow_keyboard_control)
        self._field_meta: dict[str, Any] | None = None
        self._ctx = mp.get_context("spawn")
        self._command_q: mp.Queue | None = None
        self._event_q: mp.Queue | None = None
        self._proc: mp.Process | None = None

    def start(self, port: int = 5811):
        if self._proc is not None and self._proc.is_alive():
            return
        self._command_q = self._ctx.Queue(maxsize=512)
        self._event_q = self._ctx.Queue(maxsize=32)
        self._proc = self._ctx.Process(
            target=_run_webview_process,
            args=(
                str(self.template_dir),
                bool(self.allow_keyboard_control),
                int(port),
                self._command_q,
                self._event_q,
            ),
            daemon=True,
        )
        self._proc.start()

    def poll_commands(self) -> WebMsgBuffer:
        out = WebMsgBuffer()
        if self._command_q is None:
            return out
        while True:
            try:
                cmd = self._command_q.get_nowait()
            except queue.Empty:
                break
            if not isinstance(cmd, dict):
                continue
            ct = cmd.get("type", "")
            if ct == "reset_env":
                out.reset_env = True
            elif ct == "restart_match":
                out.restart_match = True
            elif ct == "set_viewer_point":
                out.viewer_point = cmd.get("point", [3.0, 3.0, 1.0])
            elif ct == "set_viewer_look_at":
                out.viewer_look_at = cmd.get("point", [0.0, 0.0, 1.0])
            elif ct == "set_camera_preset":
                out.camera_preset = str(cmd.get("preset", "Top"))
            elif ct == "teleport_entity":
                out.teleport_cmd = (
                    str(cmd.get("name", "")),
                    float(cmd.get("x", 0.0)),
                    float(cmd.get("y", 0.0)),
                    cmd.get("z", None),
                    cmd.get("theta", None),
                )
            elif ct == "set_initial_positions":
                sp = cmd.get("spawn_points", None)
                out.spawn_points = sp if isinstance(sp, dict) else {}
            elif ct == "set_robot_velocity":
                if out.velocity_cmds is None:
                    out.velocity_cmds = []
                out.velocity_cmds.append(
                    (
                        str(cmd.get("name", "")),
                        float(cmd.get("vx", 0.0)),
                        float(cmd.get("vy", 0.0)),
                        float(cmd.get("wz", 0.0)),
                    )
                )
            elif ct == "referee_command":
                out.referee_command = str(cmd.get("command", ""))
        return out

    def emit_frame(self, rgb: np.ndarray):
        try:
            if self._event_q is None:
                return
            arr = np.ascontiguousarray(rgb)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB uint8, got shape={getattr(arr, 'shape', None)} dtype={arr.dtype}")
            image = Image.fromarray(arr, mode="RGB")
            bio = BytesIO()
            image.save(bio, format="JPEG", quality=92)
            jpeg_bytes = bio.getvalue()
            _queue_put_latest(self._event_q, {"type": "frame", "jpeg": jpeg_bytes}, max_drop=32)
        except Exception as e:
            _log_emit_throttled("emit_frame", f"[MujocoWebView] emit_frame failed: {e}")

    def emit_robot_states(self, states: dict):
        try:
            if self._event_q is None:
                return
            safe = _socket_json_safe(states)
            if isinstance(safe, dict):
                _queue_put_latest(self._event_q, {"type": "states", "states": safe}, max_drop=32)
        except Exception as e:
            _log_emit_throttled("emit_robot_states", f"[MujocoWebView] emit_robot_states failed: {e}")

    def set_field_meta(self, field_meta: dict):
        self._field_meta = _socket_json_safe(dict(field_meta))
        try:
            if self._event_q is None:
                return
            _queue_put_latest(self._event_q, {"type": "field_meta", "field_meta": dict(self._field_meta)}, max_drop=8)
        except Exception as e:
            _log_emit_throttled("field_meta", f"[MujocoWebView] field_meta broadcast failed: {e}")

    def close(self) -> None:
        try:
            if self._event_q is not None:
                _queue_put_latest(self._event_q, {"type": "shutdown"}, max_drop=1)
                try:
                    self._event_q.put_nowait(None)
                except Exception:
                    pass
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=1.5)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=0.5)
        except Exception:
            pass
