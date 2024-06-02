from PyQt5.QtWidgets import *

# 主布局
import ui
# # 子窗口实例
import window1
import window2
import window3
import window4
import window5
import window6
import window7

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        # 布局初始化
        self.ui = ui.Ui_Form()
        self.ui.setupUi(Form=self)
        # 子窗口实例化命名空间
        self.window1_one = None     # 图像几何变换
        self.window2_two = None     # 图像像素变换
        self.window3_three = None   # 图像去噪
        self.window4_four = None    # 图像锐化
        self.window5_five = None    # 边缘检测
        self.window6_six = None   # 图像分割
        self.window7_seven = None   # 卷积神经网络
        # 信号与槽定义
        self.signal_and_slot()

    def signal_and_slot(self):
        self.ui.pushButton_1.clicked.connect(self.pushButton_1)
        self.ui.pushButton_2.clicked.connect(self.pushButton_2)
        self.ui.pushButton_3.clicked.connect(self.pushButton_3)
        self.ui.pushButton_4.clicked.connect(self.pushButton_4)
        self.ui.pushButton_5.clicked.connect(self.pushButton_5)
        self.ui.pushButton_6.clicked.connect(self.pushButton_6)
        self.ui.pushButton_7.clicked.connect(self.pushButton_7)

    def pushButton_1(self):
        self.window1_one = window1.SubWindow()
        self.window1_one.show()

    def pushButton_2(self):
        self.window2_two = window2.SubWindow()
        self.window2_two.show()

    def pushButton_3(self):
        self.window3_three = window3.SubWindow()
        self.window3_three.show()

    def pushButton_4(self):
        self.window4_four = window4.SubWindow()
        self.window4_four.show()

    def pushButton_5(self):
        self.window5_five = window5.SubWindow()
        self.window5_five.show()

    def pushButton_6(self):
        self.window6_six = window6.SubWindow()
        self.window6_six.show()

    def pushButton_7(self):
        self.window7_seven = window7.SubWindow()
        self.window7_seven.show()
