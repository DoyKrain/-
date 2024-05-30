from PyQt5 import QtCore, QtGui, QtWidgets
import window
class MyApp(QtWidgets.QMainWindow, window.Ui_MainWindow):
    def __init__(self):
        super(MyApp,self).__init__(parent=None)
        self.setupUi(self)
