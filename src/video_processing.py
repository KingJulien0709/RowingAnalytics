import cv2
import torch

def predict_poses(video_path, model, conf_th=0.8, save_video=False):
  frame_rate = 30
  cap  = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file")
  else:
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
  cap.release()
  result = model.track(video_path, conf=conf_th, save=save_video, imgsz=(320, 224))
  return result, frame_rate


def get_relevant_normalized_keyframes(results, frame_rate, selected_joints, remove_wrong_numbers=False, all=False):
  normalized_keyframes_tensor  = torch.stack([torch.cat((res.keypoints.xyn[0],res.keypoints.conf[0].unsqueeze(1)), dim=1) for res in results])
  if all:
    selected_joints_keyframes = normalized_keyframes_tensor
  else:
    selected_joints_keyframes = torch.stack([n_k_t.squeeze(0)[selected_joints] for n_k_t in normalized_keyframes_tensor])
  if remove_wrong_numbers:
    error_values = torch.any(selected_joints_keyframes[:, :,0:2]==0, dim=(1,2))
    selected_joints_keyframes = selected_joints_keyframes[~error_values]
  return selected_joints_keyframes