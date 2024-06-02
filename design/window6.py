import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 子窗口布局
import window6_ui

class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window6_ui.Ui_Form()
        self.ui.setupUi(self)
        self.cv_srcImage = None
        self.ui_init()

    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_save_file.clicked.connect(self.save_image)
        self.ui.pushButton_Thread.clicked.connect(self.Thread_process)
        self.ui.pushButton_grow.clicked.connect(self.region_growing_segmentation)
        self.ui.pushButton_split.clicked.connect(self.region_split_merge_segmentation)

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

    def Thread_process(self):
        self.threshold_segmentation()

    def threshold_segmentation(self):
        # 从UI中获取一个SpinBox的阈值
        threshold_value = int(self.ui.spinBox_Thread_low.value())
        # 将源图像转换为灰度图
        gray_image = cv2.cvtColor(self.cv_srcImage.copy(), cv2.COLOR_BGR2GRAY)
        # 应用阈值分割图像
        _, segmentedImg = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 获取分割图像的尺寸
        height, width = segmentedImg.shape[0], segmentedImg.shape[1]

        # 将分割后的图像转换为QImage
        ui_image = QImage(segmentedImg.data, width, height, segmentedImg.strides[0], QImage.Format_Grayscale8)
        # 缩放图像以在UI中显示
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
        # 在UI中设置分割后的图像
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))

    def region_growing_segmentation(self):
        def region_growing(img, seed, threshold):
            rows, cols = img.shape
            segmented = np.zeros_like(img)
            visited = np.zeros_like(img)
            stack = []

            stack.append(seed)
            while stack:
                current_point = stack.pop()
                x, y = current_point
                if visited[x, y] == 1:
                    continue
                visited[x, y] = 1
                if abs(int(img[x, y]) - int(img[seed])) < threshold:
                    segmented[x, y] = img[x, y]
                    if x > 0:
                        stack.append((x - 1, y))
                    if x < rows - 1:
                        stack.append((x + 1, y))
                    if y > 0:
                        stack.append((x, y - 1))
                    if y < cols - 1:
                        stack.append((x, y + 1))
            return segmented
        threshold = self.ui.spinBox_grow_low.value()
        img = self.cv_srcImage
        height, width, _ = img.shape
        seed_point = (height // 2, width // 2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segmented_img = region_growing(img_gray, seed_point, threshold)
        ui_image = QImage(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())
        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))


    def region_split_merge_segmentation(self):
        def Division_Judge(img, h0, w0, h, w):
            area = img[h0: h0 + h, w0: w0 + w]
            mean = np.mean(area)
            std = np.std(area, ddof=1)

            total_points = area.size
            operated_points = np.sum(np.abs(area - mean) < 2 * std)

            return operated_points / total_points >= 0.95

        def Merge(img, h0, w0, h, w):
            low_threshold = int(self.ui.spinBox_split_low.value())
            high_threshold = int(self.ui.spinBox_split_high.value())

            for row in range(h0, h0 + h):
                for col in range(w0, w0 + w):
                    if low_threshold < img[row, col] < high_threshold:
                        img[row, col] = 0
                    else:
                        img[row, col] = 255

        def Recursion(img, h0, w0, h, w):
            # 如果满足分裂条件继续分裂
            if not Division_Judge(img, h0, w0, h, w) and min(h, w) > 5:
                # 递归继续判断能否继续分裂
                half_h = int(h / 2)
                half_w = int(w / 2)

                # 左上方块
                Recursion(img, h0, w0, half_h, half_w)
                # 右上方块
                Recursion(img, h0, w0 + half_w, half_h, half_w)
                # 左下方块
                Recursion(img, h0 + half_h, w0, half_h, half_w)
                # 右下方块
                Recursion(img, h0 + half_h, w0 + half_w, half_h, half_w)
            else:
                # 合并
                Merge(img, h0, w0, h, w)

        img_gray = cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2GRAY)
        print(img_gray.shape)
        hist, bins = np.histogram(img_gray, bins=256)
        segmented_img = img_gray.copy()
        Recursion(segmented_img, 0, 0, segmented_img.shape[0], segmented_img.shape[1])
        width, height = img_gray.shape
        ui_image = QImage(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
        if width > height:
            ui_image = ui_image.scaledToWidth(self.ui.label_image_2.width())
        else:
            ui_image = ui_image.scaledToHeight(self.ui.label_image_2.height())

        self.ui.label_image_2.setPixmap(QPixmap.fromImage(ui_image))
