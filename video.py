# @author: Miriam Snow
# CIS *4720 Image Processing
# Assignment 3 Road Sign Detection

import cv2
import numpy as np
import sys

video_file = sys.argv[1]
video = cv2.VideoCapture(video_file)
i = 0
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    output = frame
    #print(i)
    i = i+1
    cv2.imshow("output", output)
    cv2.waitKey(1)
    cv2.imwrite('A3/video/frame' + str(i) + '.png', output)
    # if cv2.waitKey(1) &amp; 0xFF == ord('q'):
    #     break
 
cv2.destroyAllWindows()
video.release()