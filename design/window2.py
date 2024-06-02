import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 子窗口布局
import window2_ui


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window2_ui.Ui_Form()
        self.ui.setupUi(self)
        self.ui_init()
        self.cv_srcImage = None
        self.cv_equImage = None
        self.cv_calImage = None

    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_brightness_change.clicked.connect(self.brightness_change)
        self.ui.pushButton_hist_equ.clicked.connect(self.hist_equ)
        self.ui.pushButton_gray_transform.clicked.connect(self._show_gray_image)
        self.ui.pushButton_save_file.clicked.connect(self.save_image)
        self.ui.pushButton_calcu.clicked.connect(self.open_cal)
        self.ui.pushButton_add.clicked.connect(self.add)
        self.ui.pushButton_sub.clicked.connect(self.sub)
        self.ui.pushButton_and.clicked.connect(self.andc)
        self.ui.pushButton_or.clicked.connect(self.hor)
        self.ui.pushButton_sup.clicked.connect(self.sup)
        self.ui.pushButton_not_or.clicked.connect(self.not_or)
        pass

    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '图像文件(*.jpg *.bmp *.png)')
        if file_path:
            self.cv_srcImage = cv2.imread(file_path)
            print(self.cv_srcImage.shape)
            height, width,chanels = self.cv_srcImage.shape
            ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*chanels, QImage.Format_RGB888)
            if width > height:
                ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
            else:
                ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
            self.ui.label_image_1.setPixmap(QPixmap.fromImage(ui_image))
            self._show_hist_image(flag=1)
        else:
            QMessageBox.warning(self, "导入失败", "用户失败", QMessageBox.Ok)

    def save_image(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        if save_path:
            # 保存图片
            self.ui.label_image_2.pixmap().toImage().save(save_path)
            QMessageBox.information(self, "保存成功", f"图片已成功保存到 {save_path}", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "保存失败", "用户失败", QMessageBox.Ok)

    def brightness_change(self):
        def _brightness_change(image, p=0):
            copyImage = image.copy()
            copyImage = np.array(copyImage, dtype=np.uint16)
            copyImage = copyImage + p
            copyImage = np.clip(copyImage, 0, 255)
            copyImage = np.array(copyImage, dtype=np.uint8)
            return copyImage
        self.cv_equImage = _brightness_change(image=self.cv_srcImage, p=self.ui.spinBox_brightness_change.value())
        height, width = self.cv_equImage.shape[0], self.cv_equImage.shape[1]
        ui_image = QImage(cv2.cvtColor(self.cv_equImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        self._show_hist_image(flag=2)

    def hist_equ(self):
        def histogram_equalization(image):
            # 计算图像的直方图
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])

            # 计算累积分布函数（CDF）
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            # 均衡化后的像素值
            equalized_pixels = (cdf_normalized[image] / cdf_normalized.max()) * 255

            # 将像素值限制在0到255之间
            equalized_image = np.clip(equalized_pixels, 0, 255).astype(np.uint8)
            return equalized_image
        self.cv_equImage = histogram_equalization(image=self.cv_srcImage)
        height, width,chanels = self.cv_equImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_equImage, cv2.COLOR_BGR2RGB), width, height, width*chanels,QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        self._show_hist_image(flag=2)

    def _show_hist_image(self, flag=1):
        if flag == 1:
            histImg = self._calc_gray_hist(image=self.cv_srcImage)
            width, height = histImg.shape[0], histImg.shape[1]
            ui_image = QImage(cv2.cvtColor(histImg, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
            self.ui.label_image_3.setPixmap(QPixmap.fromImage(ui_image))
        elif flag == 2:
            histImg = self._calc_gray_hist(image=self.cv_equImage)
            width, height = histImg.shape[0], histImg.shape[1]
            ui_image = QImage(cv2.cvtColor(histImg, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
            self.ui.label_image_4.setPixmap(QPixmap.fromImage(ui_image))

    def _calc_gray_hist(self, image):
        copyImage = image.copy()
        if copyImage.ndim == 3:
            copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
        histArray = cv2.calcHist([copyImage], [0], None, [256], [0, 255])  # 统计数组
        mnVal, maxVal, minLoc, macLoc = cv2.minMaxLoc(histArray)  # 找最大值
        histImg = np.zeros([256, 256, 3], np.uint8)
        hpt = int(0.9 * 256)  # 预留顶部空间
        for i in range(256):
            intensity = int(histArray[i] * hpt / maxVal)  # 柱状图高度
            cv2.line(histImg, (i, 256), (i, 256 - intensity), [255, 255, 255])  # 画线
        return histImg

    def _show_gray_image(self):
        try:
            if self.cv_srcImage is not None:
                # 获取图像高度和宽度
                height = self.cv_srcImage.shape[0]
                width = self.cv_srcImage.shape[1]
                if self.cv_srcImage.ndim == 3:
                    copyImageGray = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
                else:
                    copyImageGray = self.cv_srcImage  # 如果已经是灰度图，直接使用

                # 创建一幅图像
                result = np.zeros((height, width), np.uint8)
                # 图像灰度反色变换 s=255-r
                for i in range(height):
                    for j in range(width):
                        gray = 255 - copyImageGray[i, j]
                        result[i, j] = np.uint8(gray)

                self.cv_equImage = result
                # 创建 QImage 对象并显示在标签上
                ui_image = QImage(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB), width, height, QImage.Format_RGB888)
                # if width > height:
                #     ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
                # else:
                #     ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
                self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
                # 显示灰度图像的直方图
                self._show_hist_image(flag=2)
        except Exception as e:
            print(f"Exception: {e}")

    def open_cal(self):
        open_path, _ = QFileDialog.getOpenFileName(QFileDialog(), "导入图片", "", "*.jpg;;*.png;;All Files(*)")
        if open_path:
            # 保存图片
            QMessageBox.information(self, "导入成功", f"图片已成功导入{open_path}", QMessageBox.Ok)
            self.cv_calImage = cv2.imread(open_path)
        else:
            QMessageBox.warning(self, "导入失败", "用户失败", QMessageBox.Ok)

    def add(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                # 相加并确保结果在 0 到 255 的范围内
                ui_image[i, j] = np.clip(self.cv_srcImage[i, j] + self.cv_calImage[i, j], 0, 255)
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)


    def sub(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                # 相加并确保结果在 0 到 255 的范围内
                ui_image[i, j] = np.clip(self.cv_srcImage[i, j] - self.cv_calImage[i, j], 0, 255)
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)

    def andc(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                ui_image[i, j] = np.logical_and(self.cv_srcImage[i, j], self.cv_calImage[i, j]).astype(np.uint8) * 255
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        # if width > height:
        #     ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        # else:
        #     ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)

    def hor(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                ui_image[i, j] = np.logical_or(self.cv_srcImage[i, j], self.cv_calImage[i, j]).astype(np.uint8) * 255
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)

    def sup(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                ui_image[i, j] = 255 - self.cv_srcImage[i, j]
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)

    def not_or(self):
        height, width, channels = self.cv_srcImage.shape
        # 确保两个图像具有相同的形状
        assert self.cv_srcImage.shape == self.cv_calImage.shape, "图像形状不匹配"
        # 创建一个新的图像数组，用于存储相加的结果
        ui_image = np.zeros_like(self.cv_srcImage, dtype=np.uint8)
        print('处理中...')
        # 遍历每个像素，将对应位置的像素相加
        for i in range(self.cv_srcImage.shape[0]):
            for j in range(self.cv_srcImage.shape[1]):
                ui_image[i, j] = self.cv_srcImage[i, j] ^ self.cv_calImage[i, j]
        self.cv_equImage = ui_image
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
        # 显示灰度图像的直方图
        self._show_hist_image(flag=2)