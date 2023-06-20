# -*- coding: utf-8 -*-
import glob
import os
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from konlpy.tag import Twitter
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

import pandas as pd
from konlpy.tag import Mecab
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from adjustText import adjust_text
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QComboBox, QRadioButton, QHBoxLayout)
from geopy.geocoders import Nominatim
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QCheckBox, QFileDialog
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget, QPushButton, QFileDialog


# ============================================
# 요구사항
# ============================================
# Python을 이용한 원도우 GUI 기반 지오코딩 (주소 to 위경도 변환) 프로그램


# ============================================
# 보조
# ============================================

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0444'

# 옵션 설정
sysOpt = {
}

if (platform.system() == 'Windows'):
    globalVar['inpPath'] = './INPUT'
    globalVar['outPath'] = './OUTPUT'
    globalVar['figPath'] = './FIG'
else:
    globalVar['inpPath'] = '/DATA/INPUT'
    globalVar['outPath'] = '/DATA/OUTPUT'
    globalVar['figPath'] = '/DATA/FIG'

# 전역 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


class FileUploadWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.uploadButton = QPushButton("Upload")
        self.uploadButton.clicked.connect(self.onUpload)

        self.checkbox = QCheckBox("Enable Drag and Drop")
        self.checkbox.setChecked(True)

        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.checkbox)
        self.setLayout(self.layout)
        self.setAcceptDrops(True)

    @pyqtSlot()
    def onUpload(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options)
        if fileName:
            print(fileName)

    def dragEnterEvent(self, event):
        if self.checkbox.isChecked():
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if self.checkbox.isChecked():
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            for file in files:
                print(file)

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.api_key_entry = QLineEdit()
        self.file_entry = QLineEdit()
        self.column_combo = QComboBox()
        self.df = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('지오코딩 변환기')
        self.setAcceptDrops(True)

        # API 키 입력창
        self.api_key_entry.setPlaceholderText("API 키를 입력하세요.")
        upload_button = QPushButton('파일 업로드')
        upload_button.clicked.connect(self.upload_file)

        # 변환 버튼
        convert_button = QPushButton('변환')
        convert_button.clicked.connect(self.geocode)

        # 레이아웃 설정
        # layout = QVBoxLayout()
        # layout.addWidget(QLabel('API Key:'))
        # layout.addWidget(self.api_key_entry)
        # layout.addWidget(upload_button)
        # layout.addWidget(self.file_entry)
        # layout.addWidget(QLabel('주소 컬럼 선택:'))
        # layout.addWidget(self.column_combo)
        # layout.addWidget(convert_button)
        #
        # self.setLayout(layout)
        # self.show()

        # # 좌측 레이아웃 설정
        # mainLayout = QHBoxLayout()
        #
        # # 좌측 레이아웃에 추가할 위젯
        # leftLayout = QVBoxLayout()
        # mainLayout.addLayout(leftLayout)
        #
        # # API 키 입력창
        # self.api_key_entry.setPlaceholderText("API 키를 입력하세요.")
        # upload_button = QPushButton('파일 업로드')
        # upload_button.clicked.connect(self.upload_file)
        #
        # # 변환 버튼
        # convert_button = QPushButton('변환')
        # convert_button.clicked.connect(self.geocode)
        #
        # # 레이아웃에 위젯 추가
        # leftLayout.addWidget(QLabel('API Key:'))
        # leftLayout.addWidget(self.api_key_entry)
        # leftLayout.addWidget(upload_button)
        # leftLayout.addWidget(self.file_entry)
        # leftLayout.addWidget(QLabel('주소 컬럼 선택:'))
        # leftLayout.addWidget(self.column_combo)
        # leftLayout.addWidget(convert_button)
        #
        # self.setLayout(mainLayout)
        # self.show()


        # 레이아웃 설정
        layout = QVBoxLayout()
        leftLayout = QVBoxLayout() # 좌측 레이아웃 추가
        layout.addLayout(leftLayout) # 메인 레이아웃에 추가

        leftLayout.addWidget(QLabel('API Key:'))
        leftLayout.addWidget(self.api_key_entry)
        leftLayout.addWidget(upload_button)
        leftLayout.addWidget(self.file_entry)
        leftLayout.addWidget(QLabel('주소 컬럼 선택:'))
        leftLayout.addWidget(self.column_combo)
        leftLayout.addWidget(convert_button)

        self.setLayout(layout)
        self.show()

    # 드래그 앤 드롭 관련 이벤트 핸들러 추가
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.file_entry.setText(files[0])
        self.upload_file()

    def upload_file(self):
        if not self.file_entry.text():
            filename, _ = QFileDialog.getOpenFileName()
            self.file_entry.setText(filename)

        # 데이터 로드
        file_path = self.file_entry.text()
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)

        # 콤보 박스에 컬럼 이름 채우기
        self.column_combo.addItems(self.df.columns)

    def geocode(self):
        geolocator = GoogleV3(api_key=self.api_key_entry.text())

        # 주소 컬럼 이름
        address_column = self.column_combo.currentText()

        # 주소를 위경도로 변환
        self.df['location'] = self.df[address_column].apply(geolocator.geocode)
        self.df['point'] = self.df['location'].apply(lambda loc: tuple(loc.point) if loc else None)

        # 결과를 새 파일로 저장
        output_file, _ = QFileDialog.getSaveFileName(self, "Save file", "", "CSV Files (*.csv)")
        if output_file:
            self.df.to_csv(output_file)


# class DropArea(QtWidgets.QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAcceptDrops(True)
#
#     def dragEnterEvent(self, e):
#         m = e.mimeData()
#         if m.hasUrls():
#             e.accept()
#         else:
#             e.ignore()
#
#     def dropEvent(self, e):
#         for url in e.mimeData().urls():
#             path = url.toLocalFile()
#             if path.endswith('.csv') or path.endswith('.xlsx'):
#                 print('Uploaded: ', path)   # 파일 경로 출력
#                 # TODO: 파일 처리 코드
#                 # 파일의 확장자에 따라 다른 동작을 수행할 수 있음
#
# class LeftLayout(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.upload_button = QtWidgets.QPushButton('Upload CSV/XLSX')
#         self.upload_button.clicked.connect(self.open_file_dialog)
#
#         self.drop_area = DropArea()
#
#         self.checkbox = QtWidgets.QCheckBox('Enable drag and drop')
#         self.checkbox.toggled.connect(self.drop_area.setAcceptDrops)
#
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.addWidget(self.upload_button)
#         layout.addWidget(self.drop_area)
#         layout.addWidget(self.checkbox)
#
#     def open_file_dialog(self):
#         options = QtWidgets.QFileDialog.Options()
#         options |= QtWidgets.QFileDialog.ReadOnly
#         file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
#                                                              "CSV Files (*.csv);;XLSX Files (*.xlsx)", options=options)
#         if file_name:
#             print('Uploaded: ', file_name)  # 파일 경로 출력
#             # TODO: 파일 처리 코드
            # 업로드 된 파일 처리


# class DropArea(QtWidgets.QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAcceptDrops(True)
#
#     def dragEnterEvent(self, e):
#         m = e.mimeData()
#         if m.hasUrls():
#             e.accept()
#         else:
#             e.ignore()
#
#     def dropEvent(self, e):
#         for url in e.mimeData().urls():
#             path = url.toLocalFile()
#             if path.endswith('.csv') or path.endswith('.xlsx'):
#                 print('Uploaded: ', path)   # 파일 경로 출력
#                 # TODO: 파일 처리 코드
#                 # 파일의 확장자에 따라 다른 동작을 수행할 수 있음

# class LeftLayout(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.drop_area = DropArea()
#
#         self.checkbox = QtWidgets.QCheckBox('Enable drag and drop')
#         self.checkbox.setChecked(True)
#         self.checkbox.toggled.connect(self.toggle_drag_and_drop)
#
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.addWidget(self.drop_area)
#         layout.addWidget(self.checkbox)
#
#     def toggle_drag_and_drop(self, checked):
        self.drop_area.setAcceptDrops(checked)

# class MainWidget(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Drag and Drop")
#         self.resize(720, 480)
#         self.setAcceptDrops(True)
#
#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():
#             event.accept()
#         else:
#             event.ignore()
#
#     def dropEvent(self, event):
#         files = [u.toLocalFile() for u in event.mimeData().urls()]
#         for f in files:
#             print(f)


# class MyMainWindow(QMainWindow):
#     def __init__(self, parent=None):
#         super(MyMainWindow, self).__init__(parent)
#         self.setAcceptDrops(True)
#
#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():
#             event.accept()
#         else:
#             event.ignore()
#
#     def dropEvent(self, event):
#         files = [u.toLocalFile() for u in event.mimeData().urls()]
#         for f in files:
#             print(f)


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setWindowTitle("Multiple File Upload")

        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        add_button = QPushButton("Add Files")
        add_button.clicked.connect(self.add_files)
        layout.addWidget(add_button)

    def add_files(self):
        file_dialog = QFileDialog()
        files, _ = file_dialog.getOpenFileNames(self, "Select Files")
        if files:
            for file in files:
                self.file_list.addItem(file)


if __name__ == '__main__':
    os.environ['DISPLAY'] = 'localhost:10.0'
    display_value = os.environ.get('DISPLAY')

    app = QApplication(sys.argv)
    mainWin = MyMainWindow()
    mainWin.show()
    sys.exit(app.exec_())

    # app = QApplication(sys.argv)
    # widget = FileUploadWidget()
    # widget.show()
    # ex = MyApp()
    # sys.exit(app.exec_())

    # app = QtWidgets.QApplication(sys.argv)
    #
    # left_layout = LeftLayout()
    # left_layout.show()
    #
    # sys.exit(app.exec_())


    # app = QApplication(sys.argv)
    # ui = MainWidget()
    # ui.show()
    # sys.exit(app.exec_())