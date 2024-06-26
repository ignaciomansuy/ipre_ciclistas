import os 

# MODEL = "yolov8x.pt"
# MODEL = "runs/detect/train10/weights/best.pt"
MODEL = "runs/detect/train16/weights/best.pt"
ORIGINAL_VIDEO_NAME = "MOVI0016"
FILE_TYPE = ".avi"
SOURCE_ORIGINAL_VIDEO_PATH = os.path.join("original_videos", f"{ORIGINAL_VIDEO_NAME}{FILE_TYPE}")


COUNTER = 2
VIDEO_NAME = f"{ORIGINAL_VIDEO_NAME}-short{COUNTER}"
SOURCE_VIDEO_PATH = os.path.join("video_shorts", f"{VIDEO_NAME}{FILE_TYPE}")
COUNTER_OUTPUT = 10
TARGET_VIDEO_PATH = f"video_results/{VIDEO_NAME}-output{COUNTER_OUTPUT}{FILE_TYPE}"
TARGET_DUMMY_VIDEO_PATH = f"video_results/dummy.avi"


THICKNESS_DEFAULT = 4
TEXT_THICKNESS_DEFAULT = 4
TEXT_SCALE_DEFAULT = 2
TRACE_LENGTH_DEFAULT = 4


AMOUNT_LINE_ZONES = 8
COVERAGE_LINE_ZONES = (0.25, 0.75)