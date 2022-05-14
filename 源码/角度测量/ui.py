from tkinter import *
import tkinter.messagebox as messagebox

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.nameInput = Entry(self)
        #self.nameInput.pack() #输入对话框
        self.alertButton = Button(self, text='测量系统已开启\n \n \n提示：\n蓝色按钮==测量绳长\n红色按钮==0/90度角时自动测量绳长\n黑色按钮==测量角度\n短响==测量开始\n长响==测量结束', command=self.hello)
        self.alertButton.pack()

    def hello(self):
        #name = self.nameInput.get() or '未输入结果'
        messagebox.showinfo('Message', '测量长度 = ' )

app = Application()
# 设置窗口标题:
app.master.title('Measurement in progress')
# 主消息循环:
app.mainloop()