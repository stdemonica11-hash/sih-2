"""Live streaming endpoint.

Protocol (matches the dashboard client):
  1. Client connects to /api/v1/stream
  2. Client sends JSON:  {"type": "start", "sample_rate": 48000, "format": "f32"|"i16"}
  3. Client sends binary frames of mono PCM at that rate
  4. Every hop_sec (once >= window_sec is buffered) the server replies:
       {"type": "window", "t_sec": ..., "speech": ..., "score": ...,
        "features": {...}, "risk": {...}, "processing_ms": ..., "model": {...}}
  5. Client may send {"type": "stop"} or just disconnect; buffers are discarded.

Privacy: audio lives only in this connection's in-memory buffer.
"""
import asyncio
import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import model_info_dict, score_one_window
from app.audio.preprocess import resample
from app.config import settings
from app.inference.registry import get_detector
from app.risk_engine.engine import RiskEngine

router = APIRouter()


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    det = get_detector()
    engine = RiskEngine()

    client_sr = settings.target_sample_rate
    fmt = "f32"
    buf = np.zeros(0, dtype=np.float32)
    max_buf = int(6 * settings.target_sample_rate)
    win = int(settings.window_sec * settings.target_sample_rate)
    last_emit = 0.0
    t_session = 0.0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "detail": "Invalid JSON control message"})
                    continue
                if ctrl.get("type") == "start":
                    client_sr = int(ctrl.get("sample_rate", client_sr))
                    fmt = ctrl.get("format", "f32")
                    if not (8000 <= client_sr <= 192000) or fmt not in ("f32", "i16"):
                        await ws.send_json({"type": "error", "detail": "Bad start parameters"})
                        await ws.close(code=1008)
                        return
                    await ws.send_json({"type": "ready", "model": model_info_dict(det)})
                elif ctrl.get("type") == "stop":
                    await ws.send_json({"type": "stopped"})
                    await ws.close()
                    return
            elif msg.get("bytes") is not None:
                raw = msg["bytes"]
                if len(raw) > 2 * 1024 * 1024:      # reject absurd frames
                    await ws.send_json({"type": "error", "detail": "Frame too large"})
                    continue
                if fmt == "i16":
                    chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    chunk = np.frombuffer(raw, dtype=np.float32).copy()
                chunk, _ = resample(chunk, client_sr)
                t_session += chunk.size / settings.target_sample_rate
                buf = np.concatenate([buf, chunk])[-max_buf:]

                now = time.monotonic()
                if buf.size >= win and now - last_emit >= settings.hop_sec:
                    last_emit = now
                    seg = buf[-win:]
                    t0 = time.perf_counter()
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, score_one_window, det, engine, seg,
                        settings.target_sample_rate, t_session,
                    )
                    result["type"] = "window"
                    result["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    result["model"] = model_info_dict(det)
                    await ws.send_json(result)
    except WebSocketDisconnect:
        pass   # buffers are garbage-collected; nothing persisted