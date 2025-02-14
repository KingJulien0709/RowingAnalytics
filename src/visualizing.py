import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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