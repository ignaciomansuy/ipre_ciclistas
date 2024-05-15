
from typing import Dict, Iterable, Optional, Tuple, List
from tqdm.notebook import tqdm
import math
from VARIABLES import *
import supervision as sv
import numpy as np
import torch
from line_zone import LineZone
import csv
from ultralytics import YOLO



def calculate_hypotenuse(a, b):
  return math.sqrt(a**2 + b**2)

class VideoInfoHandler():
  def __init__(self) -> None:
    self.video_info = None
    self.va_params = {}
    self.bounding_box_annotator: sv.BoundingBoxAnnotator = None
    self.label_annotator = None
    self.byte_tracker = None
    self.line_zones: List[LineZone] = [] 
    pass
    
  def re_init(self, SOURCE_VIDEO_PATH):
    self.video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    self.__init_va_params()
    self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    self.label_annotator = sv.LabelAnnotator( text_thickness=self.va_params["text_thickness"], text_scale=self.va_params["text_scale"])
    self.byte_tracker = sv.ByteTrack(
      track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=self.video_info.fps
    )
    self.__init_line_zone_annotators()
    self.__init_line_zones()
    
  def __init_va_params(self):
    video_default_size = calculate_hypotenuse(1920, 1080)
    video_current_size = calculate_hypotenuse(self.video_info.width, self.video_info.height)
    proportion = video_current_size / video_default_size
    self.va_params = {
        "thickness": round(THICKNESS_DEFAULT * proportion),
        "text_thickness": round(TEXT_THICKNESS_DEFAULT * proportion),
        "text_scale": TEXT_SCALE_DEFAULT * proportion,
        "trace_length": round(TRACE_LENGTH_DEFAULT * proportion),
    }
  
  def __init_line_zone_annotators(self):
    self.line_zone_annotators = [sv.LineZoneAnnotator(
                thickness=1,
                text_thickness=1,
                text_scale=0.35)
              for _ in range(AMOUNT_LINE_ZONES)]
              
  def __init_line_zones(self):
    new_line_zones = self.__get_line_zones()
    if self.line_zones:
      for i, ex_line in enumerate(self.line_zones):
        new_line_zones[i].in_count = ex_line.in_count
        new_line_zones[i].out_count = ex_line.out_count
        for key, value in ex_line.class_in_count.items():
          new_line_zones[i].class_in_count[key] = value
        for key, value in ex_line.class_out_count.items():
          new_line_zones[i].class_out_count[key] = value
          
    self.line_zones = new_line_zones

  def __get_line_zones(self) -> List[LineZone]:
    line_zones = []
    for percentage in self.__divide_segment(*COVERAGE_LINE_ZONES, AMOUNT_LINE_ZONES):
        x = int(self.video_info.width * percentage)
        line_zones.append(
          LineZone(
          start=sv.Point(x, 0),
          end=sv.Point(x, self.video_info.height)
          )
        )
    return line_zones
    
  def __divide_segment(self, start, end, num_parts):
    step = (end - start) / (num_parts - 1)
    values = [start + i * step for i in range(num_parts)]
    return values
            


def callback(frame: np.ndarray, index:int, model: YOLO,
             selected_classes: List[int], vih: VideoInfoHandler) -> np.ndarray:
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
    annotated_frame=vih.label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels)
    
    annotated_frame=vih.bounding_box_annotator.annotate(
      scene=annotated_frame,
      detections=detections
    )

    for line_zone in vih.line_zones:
        line_zone.trigger(detections)
      
    for i, line_annotator in enumerate(vih.line_zone_annotators):
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter=vih.line_zones[i])

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
        ), desc=" Video processing", position=1, leave=False, total=source_video_info.total_frames - 217):
            result_frame = callback(frame, index, model, selected_classes, vih)
            sink.write_frame(frame=result_frame)

            
class LineZoneMaxCounterHelper():
  """docstring for LineZoneMaxCounterHelper."""
  def __init__(self, class_id: int, class_name: str):
    self.class_id = class_id
    self.class_name = class_name
    
    self.in_count: int = 0
    self.out_count: int = 0
    self.counting_history: List[List[str, int, int]] = []

  def update_counting(self, line_zones: List[LineZone], file_name: str):
    """Saves object counting for this file and updates instance attributes

    Args:
        line_zones (List[LineZone])
        file_name (str)
    """
    max_in, max_out = self.get_maxs(line_zones)
    
    self.counting_history.append([
      file_name,
      max_in - self.in_count,
      max_out - self.out_count
    ])
    
    self.in_count = max_in
    self.out_count = max_out
    
  def save_to_csv(self, folder_path: str):
    to_csv(
      folder_path,
      file_name=f"{self.class_name}.csv",
      data=self.counting_history
    )
    
  def get_maxs(self, line_zones: List[LineZone]):
    line_zone_max_in = max(line_zones, key=lambda x: x.class_in_count[self.class_id])
    max_in = line_zone_max_in.class_in_count[self.class_id]
    line_zone_max_out = max(line_zones, key=lambda x: x.class_out_count[self.class_id])
    max_out = line_zone_max_out.class_out_count[self.class_id]
    return max_in, max_out
      
      
def save_results(max_counters: Dict[int, LineZoneMaxCounterHelper], selected_classes: List[int], folder_path: str):
  first = True
  for class_ in selected_classes:
    # Save counting of each class in individual .csv
    counter = max_counters[class_]
    counter.save_to_csv(folder_path)
    
    if first:
      all_counts = counter.counting_history
      first = False
    else:
      for i, (_, in_, out_) in enumerate(counter.counting_history):
        all_counts[i][1] += in_
        all_counts[i][2] += out_
  
  # Save .csv with general counting for each video
  to_csv(
    folder_path,
    file_name="all_classes.csv",
    data=all_counts
  )

  # Total count for in and out direction in .txt
  save_total_count(max_counters, selected_classes, folder_path)
    


def to_csv(folder_path: str, file_name: str, data: List[List]):
  with open(os.path.join(folder_path, file_name), "w", newline="") as csv_output:
    writer = csv.writer(csv_output)
    writer.writerows(
      [["file_name", "in", "out"]] + data
    )


def save_total_count(max_counters: Dict[int, LineZoneMaxCounterHelper], selected_classes: List[int], folder_path: str):
  total_in, total_out = 0, 0
  for class_ in selected_classes:
    total_in += max_counters[class_].in_count
    total_out += max_counters[class_].out_count
  
  # TODO: change this .json and save total of echa class and sum of all classes 
  with open(os.path.join(folder_path, "total_count.txt"), 'w') as file:
    file.write(f"{total_in}, {total_out}")