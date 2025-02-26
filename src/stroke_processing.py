import numpy as np
import torch
from scipy.signal import savgol_filter

KNEE_JOINT =  torch.tensor([[1, 2],[1, 0]]) #right knee, left knee
KNEE_KP = [[12, 14, 16], [11, 13, 15]]
HIP_JOINT = torch.tensor([[1, 2],[1, 0]]) #right hip, left hip
HIP_KP = [[10, 12, 14], [9, 11, 13]]
SHOULDER_JOINT = torch.tensor([[1, 2],[1, 0]]) #right shoulder, left shoulder
SHOULDER_KP = [[6, 8, 10], [5, 7, 9]]



def calculate_angles_for_joint(keyframes, angle_pairs):
  v1_single_v = keyframes[:, angle_pairs[0, 0]] - keyframes[:, angle_pairs[0, 1]]
  v2_single_v = keyframes[:, angle_pairs[1, 0]] - keyframes[:, angle_pairs[1, 1]]
  angles = torch.acos(torch.sum(v1_single_v * v2_single_v, dim=1) / (torch.norm(v1_single_v, dim=1) * torch.norm(v2_single_v, dim=1)))*180/3.14
  return angles

#function to split rowing sequence into strokes
#using the most significant angle for stroke seperation
def extract_stroke_segments(knee_angles, framerate):
  oversmoothed_window_lenth = round(1.25*framerate)
  oversmoothing_knee_angles = savgol_filter(knee_angles, window_length=oversmoothed_window_lenth, polyorder=1) #additional smoothing for clear maxima and minima
  dx = np.diff(oversmoothing_knee_angles)
  dxx = np.diff(dx)
  turn_points = np.where(np.diff(np.sign(dx)) != 0)[0]
  maxima = turn_points[np.where(dxx[turn_points] < 0)]
  minima = turn_points[np.where(dxx[turn_points] > 0)]
  start_points = minima.tolist()
  #adjustements for beiginning and ending
  if maxima[0] < minima[0]:
    start_points.insert(0, 0)
  if minima[-1] < maxima[-1]:
    start_points.append(len(oversmoothing_knee_angles)-1)
  return np.array(start_points)


def segment_strokes(keypoints, framerate, right=True):
  knee_kp = KNEE_KP[0]
  if not right:
    knee_kp = KNEE_KP[1]
  selected_joints_keyframes = keypoints[:,knee_kp]
  knee_angles = calculate_angles_for_joint(selected_joints_keyframes, angle_pairs=KNEE_JOINT)
  start_indices = extract_stroke_segments(knee_angles,framerate) #TODO adjust function vor variable strokerate
  strokes = []
  for i in range(len(start_indices)-1):
    strokes.append(keypoints[start_indices[i]:start_indices[i+1]])
  return strokes

def get_relevant_angles(keypoints, right=True):
  '''
  Returns the angles of the knee, hip and shoulder joint for the right or left side
  '''
  #TODO: implement angle extraction for hip, shoulder and knee joint