"""
routes/stream.py
---------------------------------------------------------------------
NEW FILE. MJPEG endpoints -- a plain <img src="..."> in the browser
is enough, no JS or WebSocket needed for the video feed itself.
Simpler and lighter than binary WebSocket frames, and avoids pinning
yet another Python-3.6-compatible library version.
---------------------------------------------------------------------
"""

import time

from fastapi import APIRouter
from starlette.responses import StreamingResponse


def _mjpeg_generator(get_jpeg_fn):
    last = None
    while True:
        jpeg = get_jpeg_fn()
        # Only send when the worker has produced a NEW jpeg. The worker encodes every
        # 2nd frame, so re-sending the same bytes made the browser decode duplicates.
        if jpeg and jpeg is not last:
            last = jpeg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.02)


def build_router(workers):
    router = APIRouter()

    @router.get("/stream/raw/{eye}")
    def stream_raw(eye: str):
        worker = workers[eye]
        return StreamingResponse(_mjpeg_generator(worker.latest_raw_jpeg),
                                  media_type="multipart/x-mixed-replace; boundary=frame")

    @router.get("/stream/crop/{eye}")
    def stream_crop(eye: str):
        worker = workers[eye]
        return StreamingResponse(_mjpeg_generator(worker.latest_crop_jpeg),
                                  media_type="multipart/x-mixed-replace; boundary=frame")

    return router