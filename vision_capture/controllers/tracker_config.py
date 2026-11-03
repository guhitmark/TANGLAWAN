# tracker_config.py

TRACKING_MODE = "ai"   # "ai" = TensorRT U-Net, "ir" = legacy OpenCV, "webcam" = legacy RGB

# -- AI model (LitePupilNet) ----------------------------------
# ritnet_lite.py (Colab) -> pupil_<lite|tiny>_400x400.onnx -> build_trt_engine.py on the Jetson
ENGINE_FILE  = "pupil_lite_400x400_fp16.engine"            # built on the Jetson, lives in controllers/

TRACKER_ENABLED = True

INFER_W = 400
INFER_H = 400
ROI_TRACKING = False

# -- Pupil outline --------------------------------------------
PUPIL_FIT       = "ellipse"   # "circle" = always round | "auto" = ellipse only if nearly round | "ellipse" = old behaviour
PUPIL_MAX_RATIO = 1.4       # "auto": longest/shortest axis allowed before switching to a circle

# -- Sensor geometry ------------------------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 400
TARGET_FPS   = 30.0

FOV_DIAGONAL   = 96.0
FOV_HORIZONTAL = 80.0
FOV_VERTICAL   = 65.0

import math as _math

if TRACKING_MODE == "ai":
    # -- OV9281 / 850nm IR / 30mm WD --------------------------
    WORKING_DISTANCE_MM   = 30.0
    _rad_h = _math.radians(FOV_HORIZONTAL / 2.0)
    MM_PER_PIXEL_FALLBACK = 0.1413
    SMOOTHING_ALPHA = 0.45

elif TRACKING_MODE == "ir":
    # -- Legacy OpenCV Tracker --------------------------
    WORKING_DISTANCE_MM   = 30.0
    _rad_h = _math.radians(FOV_HORIZONTAL / 2.0)
    MM_PER_PIXEL_FALLBACK = (2.0 * WORKING_DISTANCE_MM * _math.tan(_rad_h)) / FRAME_WIDTH

    PUPIL_MIN_DIAMETER_PX = 20
    PUPIL_MAX_DIAMETER_PX = 110
    PUPIL_MIN_AREA_PX     = 314
    PUPIL_MAX_AREA_PX     = 9503

    ROI_X_FRACTION        = 0.75
    ROI_Y_FRACTION        = 0.80

    CLAHE_CLIP_LIMIT      = 3.0
    CLAHE_TILE_GRID       = 8
    BLUR_KERNEL_SIZE      = 5

    DARK_PUPIL_PERCENTILE   = 25
    BRIGHT_PUPIL_PERCENTILE = 75

    CIRCULARITY_MIN            = 0.62
    ELLIPSE_RATIO_MAX          = 1.8
    CONFIDENCE_RETRY_THRESHOLD = 0.45
    SMOOTHING_ALPHA            = 0.15

    GLINT_SUPPRESS = False

elif TRACKING_MODE == "webcam":
    # -- Legacy RGB Webcam Tracker ------------------
    WORKING_DISTANCE_MM   = 500.0
    MM_PER_PIXEL_FALLBACK = 0.9

    PUPIL_MIN_DIAMETER_PX = 30
    PUPIL_MAX_DIAMETER_PX = 200
    PUPIL_MIN_AREA_PX     = 706
    PUPIL_MAX_AREA_PX     = 31416

    ROI_X_FRACTION        = 1.0
    ROI_Y_FRACTION        = 1.0

    CLAHE_CLIP_LIMIT      = 1.5
    CLAHE_TILE_GRID       = 8
    BLUR_KERNEL_SIZE      = 7

    DARK_PUPIL_PERCENTILE   = 10
    BRIGHT_PUPIL_PERCENTILE = 90

    CIRCULARITY_MIN            = 0.60
    ELLIPSE_RATIO_MAX          = 2.2
    CONFIDENCE_RETRY_THRESHOLD = 0.50
    SMOOTHING_ALPHA            = 0.20

    GLINT_SUPPRESS = True

else:
    raise ValueError(f"Unknown TRACKING_MODE '{TRACKING_MODE}'. Use 'ai', 'ir', or 'webcam'.")

# -- Shared constants -------------------------------------------
NAN_ON_LOSS         = True
CSV_DELIMITER       = ","
CSV_FLOAT_PRECISION = 4