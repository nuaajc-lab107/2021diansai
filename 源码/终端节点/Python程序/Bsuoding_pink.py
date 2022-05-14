import cv2
import numpy as np

def nothing(value):
    pass
hsv_l = np.array([104,81,60])
hsv_u = np.array([255,255,255])
# def setup_trackbars():
#     cv2.namedWindow("Trackbars",0)
#
#     for i in ["MIN","MAX"]:
#         v =0 if i =="MIN" else 255
#
#         for j in 'HSV':
#             cv2.createTrackbar("%s_%s" % (j,i),"Trackbars",v,255,nothing)
#
# def get_trackbar_values():
#     values =[]
#
#     for i in ["MIN","MAX"]:
#         for j in 'HSV':
#             v =cv2.getTrackbarPos("%s_%s" % (j,i),"Trackbars")
#             values.append(v)
#     return values

RTSP_URL = 'http://169.254.35.15:8080/?action=stream'  # your camera's rtsp url
print("B")
#初期化USB摄像头
cap = cv2.VideoCapture(RTSP_URL)
while True:
    ret,frame =cap.read()
    if not ret:
        break
    frame =cv2.resize(frame,(480,360))
    img_hsv  =cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    # hl,sl,vl,hu,su,vu =get_trackbar_values()
    mask = cv2.inRange(img_hsv,hsv_l,hsv_u)
    #先复制一份
    mask_morph =mask.copy()
    # 函数的第一个参数表示内核的形状
    # 矩形：MORPH_RECT;
    # 交叉形：MORPH_CROSS;
    # 椭圆形：MORPH_ELLIPSE;
    # ,内核的尺寸以及锚点的位置
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # 开运算
    mask_morph =cv2.morphologyEx(mask_morph,cv2.MORPH_OPEN,kernel)
    # 闭运算
    mask_morph =cv2.morphologyEx(mask_morph,cv2.MORPH_CLOSE,kernel)
    output =cv2.bitwise_and(frame,frame,mask=mask_morph)


    # 查找图像的轮廓，返回图像的所有轮廓，从而找到所有大的联通区域，-2是取方法返回中的第二个参数
    cnts=cv2.findContours(mask_morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if cnts:
        #  取出最大的解锁边缘，解锁条件key是面积
        c =max(cnts,key=cv2.contourArea)
        # 根据最大的轮廓来读取外包圆
        ((x,y),raduis) = cv2.minEnclosingCircle(c)
        # 计算轮廓的矩
        M = cv2.moments(c)
        # 计算轮廓的重心
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # # 计算坐标
        # cx = int(M['m10'] / M['m00'])  # 求x坐标
        # cy = int(M['m01'] / M['m00'])  # 求y坐标
        # 矩形
        w, h =  25, 25
        # 只处理尺寸足够大的轮廓
        if raduis > 5:
            # # 画出最小外接圆
            # cv2.circle(frame, (int(x), int(y)), int(raduis), (0, 255, 255), 2)
            # 矩形
            cv2.rectangle(frame, (int(x)-int(raduis), int(y)-int(raduis),int(2*raduis),int(2*raduis)), color=(0,0, 255), thickness=1)  # BGR
            # 画出重心
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # 如果满足条件，就画出圆，画图函数，frame，中心，半径，颜色，厚度
        # if raduis>10:
        #     cv2.circle(frame,(int(x),int(y)),int(raduis),(0,255,0),6)
    # print("重心坐标是", cx, ",", cy, ")")
    cv2.imshow("B",frame)
    cv2.moveWindow("B",600,50)
    # cv2.imshow("mask",mask)
    # cv2.imshow("mask_morph",mask_morph)
    # cv2.imshow("output",output)

    if cv2.waitKey(300) & 0xFF is ord("q"):
        break


cap.release()
cv2.destroyAllWindows()