import cv2
import numpy as np
import time
RTSP_URLA = 'http://192.168.2.68:8080/?action=stream'  # A
RTSP_URLB = 'http://169.254.35.15:8080/?action=stream'  # B

#http://169.254.177.18:8080/?action=stream A
#http://169.254.35.15:8080/?action=stream B


capture = cv2.VideoCapture(RTSP_URLB)

#截取图片
while (True):
    # 获取一帧
    ret, frame = capture.read()

    # 将这帧转换为灰度图
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imshow('frame', gray)
    # cv2.resizeWindow("capture"
    # , 1920, 1080)

    #cv2.namedWindow("capture", cv2.WINDOW_NORMAL)  #CV_WINDOW_NORMAL
    #cv2.imshow("capture", frame)

    # cv2.moveWindow("capture", 0, 0)
    cv2.imwrite("/home/pi/diansai/B.png", frame)
    break