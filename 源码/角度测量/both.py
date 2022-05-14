import os
import jiaoduA
import jiaoduB
import math
import time
import sys

os.system("sh paizhao.sh")
time.sleep(3)

#x轴 y轴
x = jiaoduA.cx
y = jiaoduB.cx
jza = 229
jzb = 226
#求角度
a = abs(y/x)
b = (math.atan(a))
c = math.degrees(b)
print('jiaodu =',c)
# θ= F(x)

