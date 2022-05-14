import tkinter as tk  # 使用Tkinter前需要先导入

# 第1步，实例化object，建立窗口window
window = tk.Tk()

# 第2步，给窗口的可视化起名字
window.title('Measurement in progress')

# 第3步，设定窗口的大小(长 * 宽)
window.geometry('400x300+0+0')  # 这里的乘是小x

# 第4步，
#tk.Label(window, text='').pack()
tk.Label(window, text='测量系统正在运行中', bg='green').pack()  # 和前面部件分开创建和放置不同，其实可以创建和放置一步完成

# 第5步，创建一个主frame，长在主window窗口上
frame = tk.Frame(window)
frame.pack()

# 第6步，创建第二层框架frame，长在主框架frame下面
frame_l = tk.Frame(frame)  # 第二层frame，左frame，长在主frame上
#frame_r = tk.Frame(frame)  # 第二层frame，右frame，长在主frame上
frame_l.pack()
#(side='left')
#frame_r.pack(side='right')
#n蓝色按钮==测量绳长\n红色按钮==0/90度角时自动测量绳长\n黑色按钮==测量角度\n短响==测量开始\n长响==测量结束',
# 第7步，创建三组标签，为第二层frame上面的内容，分为左区域和右区域，用不同颜色标识
#tk.Label(frame_l, text='').pack()
tk.Label(frame_l, text='').pack()
tk.Label(frame_l, text='按钮提示 ：',bg='red').pack()
tk.Label(frame_l, text='蓝色按钮==测量绳长').pack()
tk.Label(frame_l, text='黑色按钮==测量角度').pack()
tk.Label(frame_l, text='红色按钮==0/90度角时自动测量绳长').pack()
#tk.Label(frame_l, text='').pack()
tk.Label(frame_l, text='声光提示 ：',bg='BlueViolet').pack()
tk.Label(frame_l, text='短响==测量开始').pack()
tk.Label(frame_l, text='长响==测量结束').pack()
tk.Label(frame_l, text='').pack()
tk.Label(frame_l, text='蓝灯==测量绳长').pack()
tk.Label(frame_l, text='绿灯==测量角度').pack()
tk.Label(frame_l, text='红灯==0/90度角时自动测量绳长').pack()
tk.Label(frame_l, text='').pack()
#tk.Label(frame_r, text='on the frame_r1', bg='yellow').pack()
#tk.Label(frame_r, text='on the frame_r2', bg='yellow').pack()
#tk.Label(frame_r, text='on the frame_r3', bg='yellow').pack()

# 第8步，主窗口循环显示
window.mainloop()