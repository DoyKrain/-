import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 子窗口布局
import window3_ui


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window3_ui.Ui_Form()
        self.ui.setupUi(self)
        self.ui_init()
        self.cv_srcImage = None
        self.cv_blurImage = None

    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_mean_blur_filter.clicked.connect(self.mean_blur_filter)
        self.ui.pushButton_median_blur_filter.clicked.connect(self.median_blur_filter)
        self.ui.pushButton_group_blur_filter.clicked.connect(self.group_blur_filter)
        self.ui.pushButton_save_file.clicked.connect(self.save_image)




    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '*.jpg *.bmp *.png *tif')
        if file_path:
            self.cv_srcImage = cv2.imread(file_path)
            height, width, channels = self.cv_srcImage.shape
            ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
            # if width > height:
            #     ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
            # else:
            #     ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
            self.ui.label_image_1.setPixmap(QPixmap.fromImage(ui_image))
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



    def mean_blur_filter(self):
        def mean_blur_filter_custom(image, size=5):
            if int(size) % 2 == 0:
                return None
            copyImage = image.copy()
            height, width, channels = copyImage.shape
            half_size = size // 2
            for i in range(half_size, height - half_size):
                for j in range(half_size, width - half_size):
                    for c in range(channels):
                        total = 0
                        for m in range(-half_size, half_size + 1):
                            for n in range(-half_size, half_size + 1):
                                total += image[i + m, j + n, c]
                        copyImage[i, j, c] = total // (size * size)
            return copyImage
        size = self.ui.spinBox_mean_ksize.value()

        self.cv_blurImage = mean_blur_filter_custom(self.cv_srcImage,size=size)
        height, width, channels = self.cv_blurImage.shape
        ui_image = QImage(self.cv_blurImage.data, width, height, width * channels, QImage.Format_RGB888)
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))

    def median_blur_filter(self):
        def median_blur_filter_custom(image, size=5):
            if int(size) % 2 == 0:
                return None

            copyImage = image.copy()
            height, width, channels = copyImage.shape

            half_size = size // 2

            for i in range(half_size, height - half_size):
                for j in range(half_size, width - half_size):
                    for c in range(channels):
                        values = []
                        for m in range(-half_size, half_size + 1):
                            for n in range(-half_size, half_size + 1):
                                values.append(image[i + m, j + n, c])
                        values.sort()
                        copyImage[i, j, c] = values[len(values) // 2]

            return copyImage

        size = self.ui.spinBox_median_ksize.value()
        self.cv_blurImage = median_blur_filter_custom(self.cv_srcImage, size=size)
        height, width, channels = self.cv_blurImage.shape
        ui_image = QImage(self.cv_blurImage.data, width, height, width * channels, QImage.Format_RGB888)

        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))

    def group_blur_filter(self):
        def combined_spatial_filter(image, size=5, iterations=1):
            copyImage = image.copy()
            height, width, channels = copyImage.shape
            half_size = size // 2

            for _ in range(iterations):
                # 均值滤波
                for i in range(half_size, height - half_size):
                    for j in range(half_size, width - half_size):
                        for c in range(channels):
                            neighborhood = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1, c]
                            copyImage[i, j, c] = int(np.mean(neighborhood))

                # 中值滤波
                for i in range(half_size, height - half_size):
                    for j in range(half_size, width - half_size):
                        for c in range(channels):
                            neighborhood = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1, c]
                            copyImage[i, j, c] = int(np.median(neighborhood))
            return copyImage

        # 获取用户输入的组合滤波器邻域大小和迭代次数
        size = self.ui.spinBox_mean_ksize.value()

        # 应用组合滤波到原始图像
        self.cv_blurImage = combined_spatial_filter(self.cv_srcImage, size=size)

        # 将处理后的图像显示在Qt GUI中
        height, width, channels = self.cv_blurImage.shape
        ui_image = QImage(self.cv_blurImage.data, width, height, width * channels, QImage.Format_RGB888)
        # if width > height:
        #     ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        # else:
        #     ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))


