import sys
import signal
import numpy as np
import cv2
import time
import ctypes
import json
import Hobot.GPIO as GPIO
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn

# ------------------------------------------------------------------
#  CHANGE ONLY THIS LINE to detect a different object
TARGET_OBJECT = "person"
# ------------------------------------------------------------------

SCORE_THRESHOLD  = 0.4
NMS_THRESHOLD    = 0.45
NMS_TOP_K        = 20
BUZZER_DURATION  = 0.3
BUZZER_PIN       = 37        # BOARD physical pin number

MODEL_PATH       = "/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin"
SENSOR_WIDTH     = 1920
SENSOR_HEIGHT    = 1080

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

is_stop = False

# ------------------------------------------------------------------
#  libpostprocess ctypes structs (same as official example)
# ------------------------------------------------------------------
class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_double),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen", ctypes.c_int),
        ("shiftData", ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",     ctypes.c_int),
        ("scaleData",    ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int * 8),
        ("numDimensions", ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",   hbDNNTensorShape_t),
        ("alignedShape", hbDNNTensorShape_t),
        ("tensorLayout", ctypes.c_int),
        ("tensorType",   ctypes.c_int),
        ("shift",        hbDNNQuantiShift_yt),
        ("scale",        hbDNNQuantiScale_t),
        ("quantiType",   ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_int),
        ("stride",       ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",     hbSysMem_t * 4),
        ("properties", hbDNNTensorProperties_t)
    ]

class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",          ctypes.c_int),
        ("width",           ctypes.c_int),
        ("ori_height",      ctypes.c_int),
        ("ori_width",       ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold",   ctypes.c_float),
        ("nms_top_k",       ctypes.c_int),
        ("is_pad_resize",   ctypes.c_int)
    ]

# Load libpostprocess
libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')
get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
get_Postprocess_result.restype  = ctypes.c_char_p

def get_TensorLayout(layout):
    return int(2) if layout == "NCHW" else int(0)

# ------------------------------------------------------------------
#  Signal handler
# ------------------------------------------------------------------
def signal_handler(sig, frame):
    global is_stop
    print("\n[INFO] Stopping...")
    is_stop = True
    sys.exit(0)

# ------------------------------------------------------------------
#  GPIO Buzzer
# ------------------------------------------------------------------
def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[INFO] Buzzer ready on BOARD pin {}".format(BUZZER_PIN))

def buzzer_beep(duration=0.3):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def cleanup_gpio():
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()

# ------------------------------------------------------------------
#  BGR -> NV12 (for image file input, same as official example)
# ------------------------------------------------------------------
def bgr2nv12(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed  = uv_planar.transpose((1, 0)).reshape((area // 2,))
    nv12 = np.zeros_like(yuv420p)
    nv12[:area]  = y
    nv12[area:]  = uv_packed
    return nv12

# ------------------------------------------------------------------
#  NV12 bytes -> BGR for display
# ------------------------------------------------------------------
def nv12_to_bgr(nv12_data, width, height):
    arr = np.frombuffer(nv12_data, dtype=np.uint8)
    arr = arr.reshape((height * 3 // 2, width))
    return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)

# ------------------------------------------------------------------
#  Run YOLOv5 postprocess using libpostprocess (official method)
# ------------------------------------------------------------------
def run_postprocess(outputs, model, input_h, input_w, orig_h, orig_w):
    info = Yolov5PostProcessInfo_t()
    info.height          = input_h
    info.width           = input_w
    info.ori_height      = orig_h
    info.ori_width       = orig_w
    info.score_threshold = SCORE_THRESHOLD
    info.nms_threshold   = NMS_THRESHOLD
    info.nms_top_k       = NMS_TOP_K
    info.is_pad_resize   = 0

    output_tensors = (hbDNNTensor_t * len(model.outputs))()

    for i in range(len(model.outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(
            outputs[i].properties.layout)

        if len(outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p)
        else:
            output_tensors[i].properties.quantiType = 2
            output_tensors[i].properties.scale.scaleData = \
                outputs[i].properties.scale_data.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float))
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_void_p)

        for j in range(len(outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = \
                outputs[i].properties.shape[j]

        libpostprocess.Yolov5doProcess(
            output_tensors[i],
            ctypes.pointer(info),
            i
        )

    result_str = get_Postprocess_result(ctypes.pointer(info))
    result_str = result_str.decode('utf-8')

    # Result string starts with "yolov5PostProcess:" (18 chars)
    try:
        data = json.loads(result_str[16:])
    except Exception:
        data = []

    return data

# ------------------------------------------------------------------
#  Draw detections, return True if target found
# ------------------------------------------------------------------
def draw_detections(frame, detections, target_label, orig_w, orig_h):
    target_found = False

    for result in detections:
        bbox  = result['bbox']
        score = float(result['score'])
        name  = result['name']

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(orig_w, int(bbox[2]))
        y2 = min(orig_h, int(bbox[3]))

        is_target = (name.lower() == target_label.lower())
        color     = (0, 0, 255) if is_target else (180, 180, 180)
        thickness = 3 if is_target else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, "{} {:.2f}".format(name, score),
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if is_target:
            target_found = True

    if target_found:
        status_text  = "TARGET: {}  |  !!! DETECTED !!!".format(target_label.upper())
        status_color = (0, 0, 255)
    else:
        status_text  = "TARGET: {}  |  NOT FOUND".format(target_label.upper())
        status_color = (0, 220, 0)

    cv2.rectangle(frame, (0, 0), (580, 36), (20, 20, 20), -1)
    cv2.putText(frame, status_text, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    return target_found

# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------
def main():
    global is_stop
    signal.signal(signal.SIGINT, signal_handler)

    target_lower = TARGET_OBJECT.strip().lower()
    if target_lower not in [c.lower() for c in COCO_CLASSES]:
        print("[WARNING] '{}' is not a COCO-80 class.".format(TARGET_OBJECT))
        print("Valid: {}".format(", ".join(COCO_CLASSES)))

    print("[INFO] Target object   : {}".format(TARGET_OBJECT))
    print("[INFO] Score threshold : {}".format(SCORE_THRESHOLD))
    print("[INFO] Model           : {}".format(MODEL_PATH))

    # Load model
    models = dnn.load(MODEL_PATH)
    model  = models[0]

    # Get model input size
    props = model.inputs[0].properties
    if props.layout == "NCHW":
        input_h, input_w = props.shape[2], props.shape[3]
    else:
        input_h, input_w = props.shape[1], props.shape[2]

    print("[INFO] Model input     : {}x{}".format(input_w, input_h))
    print("[INFO] Model loaded OK")

    # Setup buzzer
    setup_gpio()

    # Open MIPI camera - exact same call as official working example
    cam = srcampy.Camera()
    cam.open_cam(
        0,                              # camera index
        -1,                             # auto mipi
        -1,                             # auto fps
        [input_w, SENSOR_WIDTH],        # output widths
        [input_h, SENSOR_HEIGHT],       # output heights
        SENSOR_HEIGHT,
        SENSOR_WIDTH
    )
    print("[INFO] MIPI camera opened OK")
    print("[INFO] Running... Press Ctrl+C to quit.\n")

    last_beep = 0

    try:
        while not is_stop:
            # Get NV12 image at model input size from camera
            raw_img = cam.get_img(2, input_w, input_h)
            if raw_img is None:
                time.sleep(0.01)
                continue

            # BPU inference
            img_array = np.frombuffer(raw_img, dtype=np.uint8)
            outputs   = model.forward(img_array)

            # Postprocess using official libpostprocess
            detections = run_postprocess(
                outputs, model,
                input_h, input_w,
                input_h, input_w   # display same size as model input
            )

            # Convert NV12 to BGR for display
            display_frame = nv12_to_bgr(raw_img, input_w, input_h)

            # Draw and check target
            target_found = draw_detections(
                display_frame, detections, target_lower, input_w, input_h)

            # Buzzer
            now = time.time()
            if target_found and (now - last_beep) > (BUZZER_DURATION + 0.1):
                buzzer_beep(BUZZER_DURATION)
                last_beep = now
                print("[ALERT] {} detected -- BUZZER ON".format(TARGET_OBJECT.upper()))

            cv2.imshow("RDK X5 | YOLO Target Detector", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cam.close_cam()
        cv2.destroyAllWindows()
        cleanup_gpio()
        print("[INFO] Cleanup done.")


if __name__ == "__main__":
    main()