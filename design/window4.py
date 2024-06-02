import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 子窗口布局
import window4_ui


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window4_ui.Ui_Form()
        self.ui.setupUi(self)
        self.ui_init()

    def ui_init(self):
        sharpen_type_list = ['Sobel算子', 'Laplace算子']
        self.ui.comboBox_selector.addItems(sharpen_type_list)
        self.ui.comboBox_selector.activated.connect(self.comboBox_selected)
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_sobel_filter.clicked.connect(self.sobel_sharpen_filter)
        self.ui.pushButton_laplace_filter.clicked.connect(self.laplacian_sharpen_filter)
        self.ui.pushButton_save_file.clicked.connect(self.save_image)
        self.cv_srcImage = None
        self.cv_sharpenImage = None
        self._group_enable_ctrl()
        pass

    def comboBox_selected(self):
        selected = self.ui.comboBox_selector.currentText()
        self._group_enable_ctrl(selected=selected)

    def _group_enable_ctrl(self, selected=None):
        if selected is None:
            self.ui.groupBox_sobel_filter.setEnabled(False)
            self.ui.groupBox_laplace_filter.setEnabled(False)
        elif selected == 'Sobel算子':
            self.ui.groupBox_sobel_filter.setEnabled(True)
            self.ui.groupBox_laplace_filter.setEnabled(False)
        elif selected == 'Laplace算子':
            self.ui.groupBox_sobel_filter.setEnabled(False)
            self.ui.groupBox_laplace_filter.setEnabled(True)


    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '图像文件(*.jpg *.bmp *.png)')
        if file_path:
            self.cv_srcImage = cv2.imread(file_path)
            height, width, channels = self.cv_srcImage.shape
            ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
            if width > height:
                ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
            else:
                ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
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

    def sobel_sharpen_filter(self):
        def _sobel_sharpen_filter(image, mode=0):
            copyImage = image.copy()
            if copyImage.ndim == 3:
                copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
            if mode == 1:
                x = cv2.Sobel(copyImage, ddepth=cv2.CV_16S, dx=1, dy=0)
                x = cv2.convertScaleAbs(x)
                return x
            elif mode == 2:
                y = cv2.Sobel(copyImage, ddepth=cv2.CV_16S, dx=0, dy=1)
                y = cv2.convertScaleAbs(y)
                return y
            elif mode == 0:
                x = cv2.Sobel(copyImage, ddepth=cv2.CV_16S, dx=1, dy=0)
                x = cv2.convertScaleAbs(x)
                y = cv2.Sobel(copyImage, ddepth=cv2.CV_16S, dx=0, dy=1)
                y = cv2.convertScaleAbs(y)
                x_y = cv2.addWeighted(x, 0.5, y, 0.5, 0)
                return x_y
        mode = 0
        if self.ui.radioButton_sobel_dx.isChecked():
            mode = 1
        elif self.ui.radioButton_sobel_dy.isChecked():
            mode = 2
        elif self.ui.radioButton_sobel_dx_dy.isChecked():
            mode = 0
        self.cv_sharpenImage = _sobel_sharpen_filter(image=self.cv_srcImage, mode=mode)
        height, width = self.cv_sharpenImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_sharpenImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))

    def laplacian_sharpen_filter(self):
        def _laplacian_sharpen_filter(image, size=1):
            copyImage = image.copy()
            if copyImage.ndim == 3:
                copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
            copyImage = cv2.Laplacian(copyImage, ddepth=cv2.CV_16S, ksize=int(size))
            copyImage = cv2.convertScaleAbs(copyImage)
            return copyImage
        size = self.ui.spinBox_laplace_ksize.value()
        self.cv_sharpenImage = _laplacian_sharpen_filter(image=self.cv_srcImage, size=size)
        height, width = self.cv_sharpenImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_sharpenImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))