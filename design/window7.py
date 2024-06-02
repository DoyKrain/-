import cv2
from PIL.Image import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import self_built
import window7_ui
import torch
from PIL import Image
from torchvision import transforms


class SubWindow(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = window7_ui.Ui_Form()
        self.ui.setupUi(self)
        self.cv_srcImage = None
        self.path = None
        self.predic_result = None
        self.ui_init()


    def ui_init(self):
        self.ui.pushButton_open_image.clicked.connect(self.open_image)
        self.ui.pushButton_classic.clicked.connect(self.classic_network)

    def open_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(QFileDialog(), '选择图片', '', '*.jpg *.bmp *.png *tif')
        if file_path:
            self.cv_srcImage = cv2.imread(file_path)
            self.path = file_path
            height, width, channels = self.cv_srcImage.shape
            ui_image = QImage(cv2.cvtColor(self.cv_srcImage, cv2.COLOR_BGR2RGB), width, height, QImage.Format_RGB888)
            if width > height:
                ui_image = ui_image.scaledToWidth(self.ui.label_image_1.width())
            else:
                ui_image = ui_image.scaledToHeight(self.ui.label_image_1.height())
            self.ui.label_image_1.setPixmap(QPixmap.fromImage(ui_image))
        else:
            QMessageBox.warning(self, "导入失败", "用户失败", QMessageBox.Ok)



    def classic_network(self):
        # 2.定义超参数
        BATCH_SIZE = 16
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EPOCHS = 3  # 训练数据集的轮次

        # 加载模型的函数
        def load_model(model, filename='best_model4.pth'):
            model.load_state_dict(torch.load(filename))
            return model

        # 加载训练好的模型
        model = self_built.Batch_CNN(1).to(DEVICE)
        model = load_model(model, 'best_model4.pth')

        # 预处理输入图像
        def preprocess_image(image_path):
            image = Image.open(image_path).convert('L')  # 转为灰度图
            transform = transforms.Compose([
                transforms.Resize((28, 28)),  # 调整大小为模型的输入大小
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0)  # 添加一个批次维度
            return input_batch

        # 定义类别映射字典
        class_mapping = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
        }

        def test_single_image(image_path):
            model.eval()
            input_data = preprocess_image(image_path).to(DEVICE)

            with torch.no_grad():
                output = model(input_data)

            predicted_class = torch.argmax(output).item()
            predicted_digit = class_mapping[predicted_class]
            self.predic_result = predicted_digit
            print(f"The predicted digit is: {predicted_digit}")

        test_single_image(self.path)
        self.ui.textBrowser_result.setText(str(self.predic_result))


