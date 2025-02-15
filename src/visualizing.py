import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cv2

#function to plot a single stroke
def plot_single_stroke(stroke,labels,frame_rate,normalize = False, derivativ=False):
  plt.figure(figsize=(10, 8))
  for i in range(len(stroke)):
    if derivativ:
      stroke[i] = np.diff(savgol_filter(stroke[i].copy(), window_length=frame_rate//3, polyorder=1)) #use savgol filter to smooth the data before taking the derivative for beter visualization
    if normalize:
      min_value = np.min(stroke[i])
      max_value = np.max(stroke[i])
      stroke[i] = (stroke[i] - min_value) / (max_value - min_value)
      plt.plot(stroke[i])
    else:
      plt.plot(stroke[i])
  plt.legend(labels)
  plt.show()


def image_to_video(np_array, output_path, frame_rate):  
  print(np_array[0].shape)
  height, width = np_array[0].shape[1:3]
  size = (height, width)
  # Define codec (MP4 or AVI)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for .avi
  # Initialize VideoWriter
  out = cv2.VideoWriter(output_path, fourcc, frame_rate, size)
  if not out.isOpened():
      raise RuntimeError(f"Failed to open video writer. Check codec compatibility for {output_path}")
  for frame in np_array:
      frame = np.max(frame, axis=0)
      norm_heatmap = cv2.normalize(frame.T, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
      # Apply colormap to convert grayscale to BGR
      colored_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
      out.write(colored_heatmap)
  out.release()