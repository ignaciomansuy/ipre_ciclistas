
from tqdm.notebook import tqdm
import math
from VARIABLES import *
import supervision as sv
import numpy as np
import torch
from line_zone import LineZone

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
      track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=self.video_info.fps
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
          LineZone(
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
    

            


def callback(frame: np.ndarray, index:int, model, selected_classes, vih) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False, device=torch.device("cuda:0"))[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above 
    detections = detections[np.isin(detections.class_id, selected_classes)]
    # tracking detections
    detections = vih.byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = vih.trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame=vih.label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    # update line counter
    for line_zone in vih.line_zones:
        line_zone.trigger(detections)
    # return frame with box and line annotated result
    for i in range(3):
        annotated_frame = vih.line_zone_annotators[i].annotate(annotated_frame, line_counter=vih.line_zones[i])
    return  annotated_frame

def process_video(
    source_path: str,
    target_path: str,
    callback,
    model,
    selected_classes,
    vih,
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
            result_frame = callback(frame, index, model, selected_classes, vih)
            sink.write_frame(frame=result_frame)
            
