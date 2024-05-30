import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import MyApp


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp.MyApp()
    window.show()
    sys.exit(app.exec_())


