import numpy as np
import cv2



vidcap = cv2.VideoCapture('./MVI_0796.MOV')
total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
frames_step = total_frames # n
success,image = vidcap.read()
success = True
count = 0
while success:
  success,image = vidcap.read()
  if count >= 0:
  	cv2.imwrite('./кадры0/{n}.jpg'.format(n = count), image)
  print(count)
  count += 1

    # Display the resulting frame