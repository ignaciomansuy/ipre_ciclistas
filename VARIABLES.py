import os 

ORIGINAL_VIDEO_NAME = "2"
FILE_TYPE = ".mp4"
SOURCE_ORIGINAL_VIDEO_PATH = os.path.join("original_videos", f"{ORIGINAL_VIDEO_NAME}{FILE_TYPE}")


COUNTER = 2
VIDEO_NAME = f"{ORIGINAL_VIDEO_NAME}-short{COUNTER}"
SOURCE_VIDEO_PATH = os.path.join("video_shorts", f"{VIDEO_NAME}{FILE_TYPE}")
COUNTER_OUTPUT = 8
TARGET_VIDEO_PATH = f"video_results/{VIDEO_NAME}-output{COUNTER_OUTPUT}{FILE_TYPE}"
TARGET_DUMMY_VIDEO_PATH = f"video_results/dummy.avi"


THICKNESS_DEFAULT = 4
TEXT_THICKNESS_DEFAULT = 4
TEXT_SCALE_DEFAULT = 2
TRACE_LENGTH_DEFAULT = 4
