import cv2
import numpy as np


def nothing(value):
    pass


hsv_l = np.array([104,81,60])
hsv_u = np.array([255,255,255])



# if( cap.isOpened() ):
#  print("cap.isOpened() ")
# # setup_trackbars()
Y_min=[]
# for i in range(0,40):
path = '/home/pi/diansai/A.png'
frame = cv2.imread(path)

frame = cv2.resize(frame, (480, 360))
img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# hl,sl,vl,hu,su,vu =get_trackbar_values()
mask = cv2.inRange(img_hsv, hsv_l, hsv_u)
# 先复制一份
mask_morph = mask.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# 开运算
mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
# 闭运算
mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)
output = cv2.bitwise_and(frame, frame, mask=mask_morph)

# 查找图像的轮廓，返回图像的所有轮廓，从而找到所有大的联通区域，-2是取方法返回中的第二个参数
cnts = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
if cnts:
    #  取出最大的解锁边缘，解锁条件key是面积
    c = max(cnts, key=cv2.contourArea)
    # 根据最大的轮廓来读取外包圆
    ((x, y), raduis) = cv2.minEnclosingCircle(c)
    # 计算轮廓的矩
    M = cv2.moments(c)
    # 计算轮廓的重心
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # 计算坐标
    cx = int(M['m10'] / M['m00'])  # 求x坐标
    cy = int(M['m01'] / M['m00'])  # 求y坐标


    # 矩形
    w, h = 25, 25
    # 只处理尺寸足够大的轮廓
    if raduis > 5:
        # # 画出最小外接圆
        # cv2.circle(frame, (int(x), int(y)), int(raduis), (0, 255, 255), 2)
        # 矩形
        cv2.rectangle(frame, (int(x) - int(raduis), int(y) - int(raduis), int(2 * raduis), int(2 * raduis)),
                      color=(0, 0, 255), thickness=1)  # BGR
        # 画出重心
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

# cv2.imshow("original", frame)
Y_min.append(cy)


Y_min.sort(reverse=True)
#
print('Ax = ',cx)
print('Ay = ',cy)
#cv2.imshow("1",frame)
#cv2.waitKey(0)

cv2.destroyAllWindows()

# cap.release()
# cv2.destroyAllWindows()

