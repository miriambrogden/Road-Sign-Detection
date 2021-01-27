# @author: Miriam Snow
# CIS *4720 Image Processing
# Assignment 3 Road Sign Detection

import cv2

preImg = cv2.imread('A3/outputs/pre.png')
cv2.imshow('pre', preImg)
cv2.waitKey(0)
processImg = cv2.imread('A3/outputs/process.png')
cv2.imshow('process', processImg)
cv2.waitKey(0)
