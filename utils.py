
from tqdm.notebook import tqdm
import math
from VARIABLES2 import *
import supervision as sv


def calculate_hypotenuse(a, b):
  return math.sqrt(a**2 + b**2)

class VideoInfoHandler():
  def __init__(self) -> None:
    self.video_info = None
    self.va_params = {}
    self.line_zone_annotators = []
    self.label_annotator = None
    self.trace_annotator = None
    self.byte_tracker = None
    self.line_zones = [] 
    pass
    
  def re_init(self, SOURCE_VIDEO_PATH):
    self.video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    self.init_va_params()
    self.init_line_zone_annotators()
    self.label_annotator = sv.LabelAnnotator( text_thickness=self.va_params["text_thickness"], text_scale=self.va_params["text_scale"])
    self.trace_annotator = sv.TraceAnnotator(thickness=self.va_params["thickness"], trace_length=self.va_params["trace_length"])
    self.byte_tracker = sv.ByteTrack(
      track_activation_threshold=0.25, lost_track_buffer=150, minimum_matching_threshold=0.8, frame_rate=self.video_info.fps
    )
    self.init_line_zones()
    
  def init_va_params(self):
    video_default_size = calculate_hypotenuse(1920, 1080)
    video_current_size = calculate_hypotenuse(self.video_info.width, self.video_info.height)
    proportion = video_current_size / video_default_size
    self.va_params = {
        "thickness": round(THICKNESS_DEFAULT * proportion),
        "text_thickness": round(TEXT_THICKNESS_DEFAULT * proportion),
        "text_scale": TEXT_SCALE_DEFAULT * proportion,
        "trace_length": round(TRACE_LENGTH_DEFAULT * proportion),
    }
    
  def init_line_zone_annotators(self):
    self.line_zone_annotators = [sv.LineZoneAnnotator(
                thickness=self.va_params["thickness"],
                text_thickness=self.va_params["text_thickness"],
                text_scale=self.va_params["text_scale"]
                )
              for _ in range(3)]
    
  def get_line_zones(self):
    line_zones = []
    for i in [-1, 0, 1]:
        x = self.video_info.width * (1 / 2 + i * 0.15)
        line_zones.append(
          sv.LineZone(
          start=sv.Point( x, 0),
          end=sv.Point(x, self.video_info.height)
          )
        )
    return line_zones
          
  def init_line_zones(self):
    new_line_zones = self.get_line_zones()
    if self.line_zones:
      for i, ex_line in enumerate(self.line_zones):
        new_line_zones[i].in_count = ex_line.in_count
        new_line_zones[i].out_count = ex_line.out_count
    
    self.line_zones = new_line_zones
    
    

def process_video(
    source_path: str,
    target_path: str,
    callback,
    stride=1,
) -> None:
    """
    Process a video file by applying a callback function on each frame
        and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.

    Examples:
        ```python
        import supervision as sv

        def callback(scene: np.ndarray, index: int) -> np.ndarray:
            ...

        process_video(
            source_path=<SOURCE_VIDEO_PATH>,
            target_path=<TARGET_VIDEO_PATH>,
            callback=callback
        )
        ```
    """
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in tqdm(enumerate(
            sv.get_video_frames_generator(source_path=source_path, stride=stride)
        ), desc=" Video processing", position=1, leave=False, total=source_video_info.total_frames):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)