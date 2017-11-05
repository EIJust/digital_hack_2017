import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QTextEdit, QAction, QFileDialog, QPushButton, \
    QAction
from PyQt5.QtGui import QIcon, QPixmap
import pandas as pd

from granulas_detector import detect_granulas


class Gui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def open_image(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg)')
        granules = detect_granulas(file_name[0], True)
        granules[1].to_csv('granules.csv')

    def init_ui(self):
        upload_action = QAction('Upload image', self)
        upload_action.triggered.connect(self.open_image)

        self.statusBar()

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(upload_action)

        self.setGeometry(200, 200, 350, 250)
        self.setWindowTitle('Main window')


if __name__ == '__main__':
    gui = QApplication(sys.argv)

    mainWindow = Gui()
    mainWindow.show()

    ex = Gui()
    sys.exit(gui.exec_())
