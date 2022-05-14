## 本项目为2021全国大学生电子设计竞赛互联网测量

# 程序介绍

## 摄像节点 
### Linux程序
start_stream.sh	    开机创建USB 摄像头串流服务

start_browser.sh	完成基本要求的(1)、(2) 开机自动采用浏览器打开串流地址的方式显示各自节点的摄像头的视野 
## 终端节点 
### Linux程序
A.sh	调用Asuoding_pink.py挂在后台终端

B.sh	调用Bsuoding_pink.py挂在后台终端

gpio_service.sh	调用gpio.py挂在后台终端

start_processing.sh	将所有需要的服务放在此脚本内可开机自动执行

Download_chuanliu_X_5SECONDS.sh	通过FFmpeg api 将摄像节点A的视频流编码为h.264的MP4文件，时长五秒钟


### Python程序
Asuoding_pink.py	颜色识别（可修改参数）开机在终端节点显示摄像头A的网络串流数据并追踪

Bsuoding_pink.py	颜色识别（可修改参数）开机在终端节点显示摄像头B的网络串流数据并追踪

gpio.py	按钮主程序，实现按钮调用每个题目的相应的Python程序

## 角度测量
### Python程序 
ui.py ui1.py  UI界面

paizhaoA.py  终端节点A摄像头视频中的某一帧拍照

paizhaoB.py  终端节点B摄像头视频中的某一帧拍照

jiaoduA.py   终端节点A获取的某一帧拍照图片的角度

jiaoduB.py   终端节点B获取的某一帧拍照图片的角度

both.py      导入jiaoduA.py 、jiaoduB.py  测量终端节点A、B在同一帧的两个位置的角度量

最终结果仍需要进行实际测量，将both.py测量获得的数据和实际测量的数据进行曲线拟合，得到最终的函数。
即可实现通过测量。
