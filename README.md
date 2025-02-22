# RowingAnalytics
Project to analyse rowing strokes on a rowing machine by using pose estimation.

the pipeline consisists of multiple steps:
- estimation of pose keypoints with standard yolov11 pose estimation model
- segment the stroke sequence to a list of strokes, by splitring it with using the knee angles to detect the cyclical movement.

