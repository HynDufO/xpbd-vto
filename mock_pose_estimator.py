import numpy as np
import sys
import time

# Load data from the .npy file
data = np.load('start_with_t_pose.npy')

# Define frames per second (fps)
fps = 30
frame_delay = 1.0 / fps

# Delay so gpu.py finishing setting up
time.sleep(2)

# Print each frame with a delay corresponding to the fps
for p in data:
    for i in range(72):
        if i < 3:
            sys.stdout.write("0 ")
        else:
            sys.stdout.write(str(p[i]))
            if i < 71:
                sys.stdout.write(' ')
    sys.stdout.write('\n')
    sys.stdout.flush()
    time.sleep(frame_delay)
