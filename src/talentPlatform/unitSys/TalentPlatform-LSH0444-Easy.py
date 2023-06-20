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

        # 좌측 레이아웃 설정
        mainLayout = QHBoxLayout()

        # 좌측 레이아웃에 추가할 위젯
        leftLayout = QVBoxLayout()
        mainLayout.addLayout(leftLayout)

        # API 키 입력창
        self.api_key_entry.setPlaceholderText("API 키를 입력하세요.")
        upload_button = QPushButton('파일 업로드')
        upload_button.clicked.connect(self.upload_file)

        # 변환 버튼
        convert_button = QPushButton('변환')
        convert_button.clicked.connect(self.geocode)

        # 레이아웃에 위젯 추가
        leftLayout.addWidget(QLabel('API Key:'))
        leftLayout.addWidget(self.api_key_entry)
        leftLayout.addWidget(upload_button)
        leftLayout.addWidget(self.file_entry)
        leftLayout.addWidget(QLabel('주소 컬럼 선택:'))
        leftLayout.addWidget(self.column_combo)
        leftLayout.addWidget(convert_button)

        self.setLayout(mainLayout)
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


if __name__ == '__main__':
    os.environ['DISPLAY'] = 'localhost:10.0'
    display_value = os.environ.get('DISPLAY')

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())