import numpy as np
import random
import cv2

def create_demo_keypoints(t ,k_n, x_lim, y_lim):
    keypoints = np.zeros((t, k_n, 3))
    for t_i in range(t):
        for i in range(k_n):
            x = random.randint(0,x_lim)
            y = random.randint(0,y_lim)
            c = random.uniform(0.8,1)
            keypoints[t_i, i] = np.array([x,y,c])
    return keypoints


def create_gaussian_joint_map(keypoints, x_lim, y_lim, std=1):
    k_n = keypoints.shape[0] # number of keypoints
    j_map = np.zeros((k_n, x_lim, y_lim))
    for k in range(k_n):
        for i in range(x_lim):
            for j in range(y_lim):
                j_map[k,i,j] = np.exp(-((i-keypoints[k,0])**2 + (j-keypoints[k,1])**2)/(2*std**2))*keypoints[k,2]
    return j_map

def calculate_min_distace_to_limb(start, end, x0, y0):
    dx, dy = end[0]-start[0], end[1]-start[1]
    if dx == 0 and dy == 0:
        return np.sqrt((x0-start[0])**2 + (y0-start[1])**2) # if start and end are the same
    t = ((x0-start[0])*dx + (y0-start[1])*dy)/(dx**2 + dy**2) # projection of (x0,y0) on the line defined by start and end
    t = np.max([0, np.min([1, t])]) # clamp t to [0,1]
    x, y = start[0] + t*dx, start[1] + t*dy
    return np.sqrt((x-x0)**2 + (y-y0)**2)



def create_gaussian_limb_map(keypoints, x_lim, y_lim, limbs, std=1):
    l_n = len(limbs) # number of limbs
    l_map = np.zeros((l_n, x_lim, y_lim))
    for l in range(l_n):
        start, end = keypoints[limbs[l][0]], keypoints[limbs[l][1]]
        for i in range(x_lim):
            for j in range(y_lim):
                l_map[l,i,j] = np.exp(-calculate_min_distace_to_limb(start, end, i, j)**2/(2*std**2))*np.min([start[2], end[2]])
    return l_map


def create_max_pooling_map(map):
    return np.max(map, axis=0)

def create_bins(t, T):
    bins = np.full(T, t//T)
    bins[:t%T] += 1
    random.shuffle(bins)
    return bins

# keypoints: (t, K, 3) -> (T, K, 3)  assume t < T, T is fixed with 48, t is the number of frames per stroke (30-60 fps and strokerate 15-40 spm)
def uniform_sample_from_frames(keypoints, T):
    t = keypoints.shape[0]
    bins = create_bins(t, T)
    sampled_keypoints = np.zeros((T, keypoints.shape[1], keypoints.shape[2]))
    count = 0
    for i in range(T):
        sampled_index = count + np.random.randint(0, bins[i])
        sampled_keypoints[i] = keypoints[sampled_index]
        count = count + bins[i]
    return sampled_keypoints


#zoom to region of interest T x K x 3 -> T x K x 3 but only the region of interest
def zoom_to_roi(map,roi, border_padding=0):
    print(map.shape)
    print(roi[0])
    croped_map = map[:, int(roi[0]+border_padding):int(roi[1]+border_padding), int(roi[2]+border_padding):int(roi[3]+border_padding)]
    print(croped_map.shape)
    resized_map = np.array([cv2.resize(croped_map[i], (56,56)) for i in range(map.shape[0])])
    print(resized_map.shape)
    return resized_map