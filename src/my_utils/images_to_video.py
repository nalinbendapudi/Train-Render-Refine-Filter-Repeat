import cv2
import os

# 1. Load, sort, and read images
folder = 'examples/single_img_results/mcgs_fav_imgs'
files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
frame = cv2.imread(os.path.join(folder, files[0]))
h, w, l = frame.shape

num_files = len(files)
video_length = 28 #seconds
frame_rate = num_files / video_length

# 2. Setup video writer
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))

# 3. Add frames
for f in files:
    video.write(cv2.imread(os.path.join(folder, f)))

video.release()
