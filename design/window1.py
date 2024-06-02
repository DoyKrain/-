import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import window1_ui


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window1_ui.Ui_Form()
        self.ui.setupUi(self)
        self.ui_init()
        self.zoom_factor = 1.0
        self.cv_srcImage = None
        self.q_image = None

    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_save_file.clicked.connect(self.save_file)
        self.ui.pushButton_zoom_in.clicked.connect(self.zoom_in)
        self.ui.pushButton_zoom_out.clicked.connect(self.zoom_out)
        self.ui.pushButton_zoom_reset.clicked.connect(self.zoom_reset)
        self.ui.pushButton_screenshot.clicked.connect(self.clip_image)
        self.ui.pushButton_spin.clicked.connect(self._center_spin_control)
        self.ui.pushButton_updownX.clicked.connect(self._updownX_control)
        self.ui.pushButton_updownY.clicked.connect(self._updownY_control)
        self.ui.pushButton_updownXY.clicked.connect(self._updownXY_control)
        self.ui.pushButton_move.clicked.connect(self._move_control)


    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '图像文件(*.jpg *.bmp *.png)')
        self.cv_srcImage = cv2.imread(file_path)
        height, width, channels = self.cv_srcImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self.zoom_factor = 1.0
        self._show_qimage_to_label(ui_image)

    def save_file(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        if save_path:
            # 保存图片
            self.ui.label_image.pixmap().toImage().save(save_path)
            QMessageBox.information(self, "保存成功", f"图片已成功保存到 {save_path}", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "保存失败", "用户失败", QMessageBox.Ok)


    def zoom_in(self):
        self.zoom_factor += 0.1
        height, width, channels = self.cv_srcImage.shape
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)

        ui_image = ui_image.scaled(new_width, new_height)

        self._show_qimage_to_label(ui_image)

    def zoom_out(self):
        self.zoom_factor -= 0.1
        height, width, channels = self.cv_srcImage.shape
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        ui_image = ui_image.scaled(new_width, new_height)
        self._show_qimage_to_label(ui_image)

    def zoom_reset(self):
        self.zoom_factor = 1.0
        height, width, channels = self.cv_srcImage.shape
        ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)

    def clip_image(self):
        anchor_x = int(self.ui.spinBox_anchor_x.value())
        anchor_y = int(self.ui.spinBox_anchor_y.value())
        offset_x = int(self.ui.spinBox_X_offset.value())
        offset_y = int(self.ui.spinBox_Y_offset.value())
        clip_image = self.cv_srcImage.copy()[anchor_y: offset_y - 1, anchor_x: offset_x - 1]
        cv2.imshow('clip_image', clip_image)
        cv2.waitKey(0)

    def _show_zoom_factor(self):
        self.ui.label_zoom_factor_2.setText(str(self.zoom_factor)[:3] + 'x')

    def _update_srcImage_size(self):
        height, width, channels = self.cv_srcImage.shape
        self.ui.label_srcImage_size.setText('原图X轴*Y轴：' + str(width) + ' x ' + str(height))
        self.ui.spinBox_anchor_x.setMaximum(width)
        self.ui.spinBox_anchor_y.setMaximum(height)
        self.ui.spinBox_X_offset.setMaximum(width)
        self.ui.spinBox_Y_offset.setMaximum(height)
        self.ui.spinBox_anchor_x.setValue(0)
        self.ui.spinBox_anchor_y.setValue(0)
        self.ui.spinBox_X_offset.setValue(width)
        self.ui.spinBox_Y_offset.setValue(height)

    def _center_spin_control(self):
        angle = int(self.ui.spinBox_spin.value())
        height, width, channels = self.cv_srcImage.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(self.cv_srcImage, rotation_matrix, (width, height))
        ui_image = QImage(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)

    def _updownX_control(self):
        height, width, channels = self.cv_srcImage.shape
        ui_image = np.zeros_like(self.cv_srcImage)
        for i in range(height):
            ui_image[i, :, :] = self.cv_srcImage[height - i - 1, :, :]
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)

    def _updownY_control(self):
        height, width, channels = self.cv_srcImage.shape
        print(self.cv_srcImage.shape)
        ui_image = np.zeros_like(self.cv_srcImage)
        for j in range(width):
            ui_image[:, j, :] = self.cv_srcImage[:, width - j - 1, :]
        print(ui_image.shape)
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)

    def _updownXY_control(self):
        height, width, channels = self.cv_srcImage.shape
        ui_image = self.cv_srcImage[::-1, ::-1, :]
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), width, height,width*channels, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)

    def _move_control(self):
        spin_02 = int(self.ui.spinBox_02.value())
        spin_12 = int(self.ui.spinBox_12.value())
        width, height, channels = self.cv_srcImage.shape
        # 打印调试信息
        print("原始平移值 spin_02:", spin_02, "spin_12:", spin_12)
        M = np.float32([[1, 0, spin_02], [0, 1, spin_12]])
        print("变换矩阵 (M):", M)
        ui_image = self.cv_srcImage.copy()
        for i in range(channels):
            ui_image[:, :, i] = cv2.warpAffine(ui_image[:, :, i], M, (height, width))
        ui_image = QImage(cv2.cvtColor(ui_image, cv2.COLOR_BGR2RGB), height, width, QImage.Format_RGB888)
        self._show_qimage_to_label(ui_image)
        # 打印额外的调试信息
        print("原始图像形状:", self.cv_srcImage.shape)
        print("变换后图像形状:", ui_image.width(), ui_image.height())

    def _show_qimage_to_label(self, qimage):
        self.ui.label_image.setPixmap(QPixmap.fromImage(qimage))
        self._show_zoom_factor()
        self._update_srcImage_size()



