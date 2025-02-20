import cv2
import torch
import numpy as np

def predict_poses(video_path, model, conf_th=0.5, save_video=False):
  frame_rate = 30
  cap  = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file")
  else:
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
  cap.release()
  result = model.track(video_path, conf=conf_th, save=save_video, imgsz=(320, 320))
  return result, frame_rate


def clean_keypoints(keypoints):
    mask = torch.all(keypoints[..., :2] == 0, dim=-1)  # Check if both x and y are 0
    keypoints[mask] = keypoints.new_tensor([0.5, 0.5, 0.0])  
    # replacement of wrong values centered in the image for not influencing the roi cropped heatmap, with a confidence of 0
    return keypoints
    

def get_relevant_normalized_keyframes(results, frame_rate, selected_joints, clean_numbers=False, all=False):
  normalized_keyframes_tensor  = torch.stack([torch.cat((res.keypoints.xyn[0],res.keypoints.conf[0].unsqueeze(1)), dim=1) for res in results])
  if all:
    selected_joints_keyframes = normalized_keyframes_tensor
  else:
    selected_joints_keyframes = torch.stack([n_k_t.squeeze(0)[selected_joints] for n_k_t in normalized_keyframes_tensor])
  if clean_numbers:
    selected_joints_keyframes = clean_keypoints(selected_joints_keyframes)
  return selected_joints_keyframes

