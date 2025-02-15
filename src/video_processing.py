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


def image_to_video(np_array, output_path, frame_rate):  
  print(np_array[0].shape)
  height, width = np_array[0].shape[1:3]
  size = (width, height)
  # Define codec (MP4 or AVI)
  fourcc = cv2.VideoWriter_fourcc(*'avi1')  # Use 'XVID' for .avi

  # Initialize VideoWriter
  out = cv2.VideoWriter(output_path, fourcc, frame_rate, size)

  if not out.isOpened():
      raise RuntimeError(f"Failed to open video writer. Check codec compatibility for {output_path}")

  for frame in np_array:
      # Normalize heatmap (0 to 255)
      frame = np.max(frame, axis=0)

      norm_heatmap = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

      # Apply colormap to convert grayscale to BGR
      colored_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

      # Write frame to video
      out.write(colored_heatmap)

    # Release video writer
  out.release()