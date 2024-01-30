import numpy as np
import math
import cv2

def get_background(raw_frames, number_of_frames=50):
    frame_indices = len(raw_frames) * np.random.uniform(size=number_of_frames)

    frames = []
    for rawIdx in frame_indices:
        idx = math.floor(rawIdx)
        frames.append(raw_frames[idx])
    
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    return blurred