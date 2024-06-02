import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 子窗口布局
import window5_ui


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window5_ui.Ui_Form()
        self.ui.setupUi(self)
        self.cv_srcImage = None
        self.ui_init()

    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_canny.clicked.connect(self.canny_process)
        self.ui.pushButton_sobel.clicked.connect(self.sobel_process)
        self.ui.pushButton_prewitt.clicked.connect(self.prewitt_process)

    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '图像文件(*.jpg *.bmp *.png)')
        self.cv_srcImage = cv2.imread(file_path)
        height, width = self.cv_srcImage.shape[0], self.cv_srcImage.shape[1]
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
        self.ui.label_image_1.setPixmap(QPixmap.fromImage(ui_image))

    def save_image(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        if save_path:
            # 保存图片
            self.ui.label_image_2.pixmap().toImage().save(save_path)
            QMessageBox.information(self, "保存成功", f"图片已成功保存到 {save_path}", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "保存失败", "用户失败", QMessageBox.Ok)

    def canny_process(self):
        low_th = int(self.ui.spinBox_canny_low_th.value())
        high_th = int(self.ui.spinBox_canny_high_th.value())
        edgeImg = cv2.Canny(self.cv_srcImage.copy(), low_th, high_th)
        height, width = edgeImg.shape[0], edgeImg.shape[1]
        ui_image = QImage(cv2.cvtColor(edgeImg, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))

    def sobel_process(self):
        # 将图像转换为灰度图
        gray_img = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        # 应用Sobel算子获取水平和垂直方向的梯度
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度的幅度
        gradient_mag = cv2.magnitude(sobel_x, sobel_y)
        # 将幅度转换为8位灰度图像
        gradient_mag = cv2.convertScaleAbs(gradient_mag)
        # 获取图像的高度和宽度
        height, width = gradient_mag.shape[0], gradient_mag.shape[1]
        # 将图像转换为Qt可用的QImage格式
        ui_image = QImage(gradient_mag.data, width, height, gradient_mag.strides[0], QImage.Format_Grayscale8)
        # 根据宽高比例缩放图像以适应标签大小
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
        # 在Qt界面上显示图像
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))


    def prewitt_process(self):
        # 将图像转换为灰度图
        gray_img = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)

        # 定义Prewitt算子的核
        kernel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
        kernel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)

        # 应用Prewitt算子获取水平和垂直方向的梯度
        prewitt_x = cv2.filter2D(gray_img, cv2.CV_64F, kernel_x[0] * kernel_x[1].T)
        prewitt_y = cv2.filter2D(gray_img, cv2.CV_64F, kernel_y[0] * kernel_y[1].T)

        # 计算梯度的幅度
        gradient_mag = cv2.magnitude(prewitt_x, prewitt_y)

        # 将幅度转换为8位灰度图像
        gradient_mag = cv2.convertScaleAbs(gradient_mag)

        # 获取图像的高度和宽度
        height, width = gradient_mag.shape[0], gradient_mag.shape[1]

        # 将图像转换为Qt可用的QImage格式
        ui_image = QImage(gradient_mag.data, width, height, gradient_mag.strides[0], QImage.Format_Grayscale8)

        # 根据宽高比例缩放图像以适应标签大小
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())

        # 在Qt界面上显示图像
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))