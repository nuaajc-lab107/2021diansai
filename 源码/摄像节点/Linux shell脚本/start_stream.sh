#!/bin/bash
nohup /home/pi/diansai/mjpg-streamer-master/mjpg-streamer-experimental/mjpg_streamer -i "/home/pi/diansai/mjpg-streamer-master/mjpg-streamer-experimental/input_uvc.so -n -f 30 -r 1920x1080" -o "/home/pi/diansai/mjpg-streamer-master/mjpg-streamer-experimental/output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www"
