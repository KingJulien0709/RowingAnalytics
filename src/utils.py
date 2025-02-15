import pickle
import os

#read a file with path and labels of the strokes per video and safe all of them in a pickle file
PICKLE_NAME = "estimated_pose_strokes.pickle"

def save_pose_to_pickle(keypoints,labels, video_id):
  with open(PICKLE_NAME, "wb") as f:
    pickle.dump({video_id: {"keypoints": keypoints, "labels": labels}}, f)
    f.close()

def read_pose_from_pickle(video_id):
  with open(PICKLE_NAME, "rb") as f:
    data = pickle.load(f)
    f.close()
  return data[video_id]

def read_pose_from_pickle_all():
  with open(PICKLE_NAME, "rb") as f:
    data = pickle.load(f)
    f.close()
  return data
