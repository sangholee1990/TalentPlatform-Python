# -*- coding: utf-8 -*-
import glob
import os
import platform
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
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
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import googlemaps
import pandas as pd
import qtmodern.styles
import qtmodern.windows

import zipfile
import os
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import pytz

# ============================================
# 요구사항
# ============================================
# Python을 이용한 원도우 GUI 기반 지오코딩 (주소 to 위경도 변환) 프로그램


# ============================================
# 보조
# ============================================
def get_encoding(file):
    # 바이너리로 파일을 열어 인코딩을 추정한다.
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0444'

# 옵션 설정
sysOpt = {
    # 구글 API 정보
    # , 'endList' : ['EUC-KR', 'UTF-8', 'CP949']
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
# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['axes.unicode_minus'] = False


# 메인 윈도우 클래스 정의
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 전체 선택 상태를 추적하는 변수 추가
        self.select_all_status = True
        self.select_all_status2 = True
        self.initUI()

    def initUI(self):
        # 윈도우 타이틀 및 아이콘 설정
        self.setWindowTitle('PyQt5 원도우 GUI 기반 지오코딩 프로그램')
        self.setWindowIcon(QIcon('icon.png'))

        # 그리드 레이아웃 생성
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

        self.search_label = QLabel('(선택) 인증키')
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('없을 시 기본값 설정')
        self.column_label = QLabel('(필수) 주소 컬럼')
        self.column_combo = QComboBox()
        self.column_combo.addItem('선택')

        grid.addWidget(self.search_label, 0, 0, alignment=Qt.AlignCenter)
        grid.addWidget(self.search_edit, 0, 1)
        grid.addWidget(self.column_label, 0, 2, alignment=Qt.AlignCenter)
        grid.addWidget(self.column_combo, 0, 3)

        self.setLayout(grid)

        # 대상 파일 영역 위젯 생성 및 배치
        self.upload_button = QPushButton('파일 업로드')
        self.upload_button.clicked.connect(self.upload_files)
        self.convert_button = QPushButton('위경도 변환')
        self.convert_button.clicked.connect(self.convert_files)
        self.delete_button = QPushButton('삭제')
        self.delete_button.clicked.connect(self.delete_files)
        self.select_all_button = QPushButton('전체 선택')
        self.select_all_button.clicked.connect(self.select_all_files)
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(['선택', '파일명'])
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        grid.addWidget(self.select_all_button, 1, 0)
        grid.addWidget(self.upload_button, 1, 1)
        grid.addWidget(self.convert_button, 1, 2)
        grid.addWidget(self.delete_button, 1, 3)
        grid.addWidget(self.file_table, 2, 0, 1, 4)

        # 변환 파일 영역 위젯 생성 및 배치
        self.select_all_button2 = QPushButton('전체 선택')
        self.select_all_button2.clicked.connect(self.select_all_files2)
        self.download_button = QPushButton('다운로드')
        self.download_button.clicked.connect(self.download_files)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(['선택', '파일명'])
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        grid.addWidget(self.select_all_button2, 3, 0)
        grid.addWidget(self.download_button, 3, 1)
        grid.addWidget(self.result_table, 4, 0, 1, 4)

        # 폰트 설정
        font = QFont("Arial", 12)
        self.setFont(font)

        # 그리드 레이아웃 간격 조정
        # grid.setHorizontalSpacing(20)
        # grid.setVerticalSpacing(10)

        # 윈도우 크기 및 위치 조정
        self.resize(1000, 800)
        # self.center()

        # 윈도우 보이기
        self.show()

    def center(self):
        # 윈도우를 화면 가운데로 이동하는 메소드
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def upload_files(self):
        # 파일 업로드 버튼을 눌렀을 때 실행되는 메소드
        # 파일 다이얼로그를 통해 csv/xlsx 파일을 선택하고 테이블에 추가한다.
        # 파일이 이미 존재하면 중복 검사를 한다.
        files, _ = QFileDialog.getOpenFileNames(self, '파일 업로드', '', 'CSV files (*.csv);;Excel files (*.xlsx)')

        # 콤보박스를 초기화한다.
        self.column_combo.clear()

        columns_set = set()  # 중복되지 않는 컬럼명을 저장할 set
        for file in files:
            filename = file.split('/')[-1]
            if self.check_duplicate(filename): continue

            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 1, QTableWidgetItem(filename))

            check = QCheckBox()
            check.setChecked(True)
            check.stateChanged.connect(self.check_state_changed)

            # Use QWidget to hold checkbox and center it within cell.
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(check)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.file_table.setCellWidget(row, 0, widget)

            # check = QCheckBox()
            # check.setChecked(True)
            # check.stateChanged.connect(self.check_state_changed)
            # self.file_table.setCellWidget(row, 0, check)

            # 초기 데이터프레임 설정
            df = pd.DataFrame()

            # 파일의 확장자를 체크하고 알맞은 pandas 함수를 이용해 파일을 읽는다.
            extension = filename.split('.')[-1]
            # encoding = get_encoding(file)
            if extension == 'csv':
                encList = ['EUC-KR', 'UTF-8', 'CP949']
                for enc in encList:
                    try:
                        df = pd.read_csv(file, encoding=enc)
                        break
                    except Exception as e:
                        continue
            elif extension == 'xlsx':
                df = pd.read_excel(file)
            else:
                continue

            # 데이터프레임의 컬럼명을 set에 추가한다.
            columns_set.update(df.columns)

        # set의 모든 요소를 콤보박스에 추가한다.
        self.column_combo.addItems(list(columns_set))

    def check_duplicate(self, filename):
        # 파일 중복 검사를 하는 메소드
        # 테이블에 이미 존재하는 파일명과 비교하고 결과를 반환한다.
        for i in range(self.file_table.rowCount()):
            if filename == self.file_table.item(i, 1).text():
                return True
        return False

    def check_duplicate2(self, filename):
        # 파일 중복 검사를 하는 메소드
        # 테이블에 이미 존재하는 파일명과 비교하고 결과를 반환한다.
        for i in range(self.result_table.rowCount()):
            if filename == self.result_table.item(i, 1).text():
                return True
        return False

    def convert_files(self):
        # 위경도 변환 버튼을 눌렀을 때 실행되는 메소드
        # 구글 API 인증키를 입력하거나 기본값을 사용하고 컬럼을 선택한다.
        # 체크된 파일들에 대해 지오코딩을 수행하고 결과를 테이블에 추가한다.

        checked_files = [self.file_table.cellWidget(i, 0).layout().itemAt(0).widget().isChecked() for i in range(self.file_table.rowCount()) if isinstance(self.file_table.cellWidget(i, 0).layout().itemAt(0).widget(), QCheckBox)]
        print(checked_files)

        if not any(checked_files):
            self.show_toast_message('대상 파일을 선택해 주세요.')
            return False

        selected_column = self.column_combo.currentText()
        if not selected_column or selected_column == '선택':
            self.show_toast_message('주소 컬럼을 선택해 주세요.')
            return False

        key = self.search_edit.text()
        if not key:
            key = sysOpt['googleApiKey']
        print(f'[CHECK] key : {key}')

        try:
            gmaps = googlemaps.Client(key=key)
        except Exception as e:
            self.show_toast_message('구글 API키를 인증해 주세요.')
            return False

        selected_column = self.column_combo.currentText()

        for i in range(self.file_table.rowCount()):
            # check = self.file_table.cellWidget(i, 0)
            widget = self.file_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            filename = self.file_table.item(i, 1).text()
            fileNameNoExt = os.path.basename(filename).split('.')[0]

            if self.check_duplicate2(filename): continue

            extension = filename.split('.')[-1]

            if extension == 'csv':
                encList = ['EUC-KR', 'UTF-8', 'CP949']
                for enc in encList:
                    try:
                        df = pd.read_csv(filename, encoding=enc)
                        break
                    except Exception as e:
                        continue

            if extension == 'xlsx':
                df = pd.read_excel(filename)

            # 구글 위경도 변환
            addrList = set(df[selected_column])

            matData = pd.DataFrame()
            for j, addrInfo in enumerate(addrList):
                print(f'[CHECK] [{round((i / len(addrList)) * 100.0, 2)}] addrInfo : {addrInfo}')
                self.show_toast_message(f'[{round((i / len(addrList)) * 100.0, 2)}] {fileNameNoExt}')

                # 초기값 설정
                matData.loc[j, selected_column] = addrInfo
                matData.loc[j, '위도'] = None
                matData.loc[j, '경도'] = None

                try:
                    rtnGeo = gmaps.geocode(addrInfo, language='ko')
                    if (len(rtnGeo) < 1): continue

                    # 위/경도 반환
                    matData.loc[j, '위도'] = rtnGeo[0]['geometry']['location']['lat']
                    matData.loc[j, '경도'] = rtnGeo[0]['geometry']['location']['lng']

                except Exception as e:
                    print(f"Exception : {e}")

            # addr를 기준으로 병합
            df = df.merge(matData, left_on=[selected_column], right_on=[selected_column], how='inner')

            saveFile = f'{fileNameNoExt}_위경도 변환.csv'

            # 파일 저장
            df.to_csv(saveFile, index=False)

            # [변환 목록]에 행 추가
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 1, QTableWidgetItem(saveFile))

            check = QCheckBox()
            check.setChecked(True)
            check.stateChanged.connect(self.check_state_changed2)

            # Use QWidget to hold checkbox and center it within cell.
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(check)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.result_table.setCellWidget(row, 0, widget)


    # [대상 목록]에서 [삭제] 기능
    def delete_files(self):
        # 삭제 버튼을 눌렀을 때 실행되는 메소드
        # 체크된 파일들을 테이블에서 삭제한다.
        rows = []
        for i in range(self.file_table.rowCount()):
            # check = self.file_table.cellWidget(i, 0)
            widget = self.file_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            rows.append(i)

        rows.reverse()  # 역순으로 삭제해야 올바르게 동작함

        for row in rows:
            self.file_table.removeRow(row)
            self.show_toast_message('삭제')

    # [대상 목록]에서 [전체 선택] 기능
    def select_all_files(self):
        # 전체 선택 버튼을 눌렀을 때 실행되는 메소드
        # 업로드 대상 파일을 전체 선택하거나 해제한다.
        self.select_all_status = not self.select_all_status  # 상태를 반전시킨다.
        for i in range(self.file_table.rowCount()):
            # check = self.file_table.cellWidget(i, 0)
            # check.setChecked(self.select_all_status)  # 상태에 따라 체크박스를 설정한다.

            widget = self.file_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()
            check.setChecked(self.select_all_status)

    # [변환 목록]에서 [전체 선택] 기능
    def select_all_files2(self):
        # 전체 선택 버튼2를 눌렀을 때 실행되는 메소드
        # 변환 파일을 전체 선택하거나 해제한다.
        self.select_all_status2 = not self.select_all_status2  # 상태를 반전시킨다.
        for i in range(self.result_table.rowCount()):
            widget = self.result_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()
            check.setChecked(self.select_all_status2)

    # [변환 목록]에서 [다운로드] 기능
    def download_files(self):
        # 다운로드 버튼을 눌렀을 때 실행되는 메소드
        # 체크된 변환 파일들을 다운로드한다.
        rows = []
        for i in range(self.result_table.rowCount()):
            widget = self.result_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()
            if not check.isChecked(): continue
            rows.append(i)

        if len(rows) < 1:
            self.show_toast_message('다운로드 파일을 선택해 주세요.')
            return False

        # for row in rows:
        #     filename = self.result_table.item(row, 1).text()
            # self.show_toast_message(f"{filename} 파일 다운로드 중")

        # Choose where to save the zip file
        # zip_path, _ = QFileDialog.getSaveFileName(self, "Save Zip", "", "ZIP files (*.zip)")
        zipFile = f'{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")}_다운로드.zip'
        with zipfile.ZipFile(zipFile, "w") as zipf:
            for row in rows:
                filename = self.result_table.item(row, 1).text()
                csv_path = os.path.join(os.getcwd(), filename)
                zipf.write(csv_path, arcname=os.path.basename(csv_path))

            if (os.path.exists(zipFile)):
                self.show_toast_message("다운로드 완료")
            else:
                self.show_toast_message("다운로드 실패")


    def show_toast_message(self, message):
        # Toast 메시지를 보여주는 메소드
        toast = QLabel(message, self)
        toast.setStyleSheet("background-color:#333;color:#fff;padding:8px;border-radius:4px;")
        toast.setAlignment(Qt.AlignCenter)
        toast.setGeometry(10, 10, toast.sizeHint().width(), toast.sizeHint().height())
        toast.show()
        toast.raise_()
        # QTimer.singleShot(3000, toast.close)
        QTimer.singleShot(1000, toast.close)

    def check_state_changed(self):
        # 체크 박스의 상태가 변경될 때 실행되는 메소드
        for i in range(self.file_table.rowCount()):
            # checkbox = self.file_table.cellWidget(i, 0)
            widget = self.file_table.cellWidget(i, 0)
            checkbox = widget.layout().itemAt(0).widget()

            if checkbox.isChecked():
                self.show_toast_message('선택')
            else:
                self.show_toast_message('해제')

    def check_state_changed2(self):
        # 체크 박스의 상태가 변경될 때 실행되는 메소드
        for i in range(self.result_table.rowCount()):
            # checkbox = self.file_table.cellWidget(i, 0)
            widget = self.result_table.cellWidget(i, 0)
            checkbox = widget.layout().itemAt(0).widget()

            if checkbox.isChecked():
                self.show_toast_message('선택')
            else:
                self.show_toast_message('해제')


if __name__ == '__main__':
    os.environ['DISPLAY'] = 'localhost:10.0'
    display_value = os.environ.get('DISPLAY')

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mw = qtmodern.windows.ModernWindow(mainWindow)
    mw.show()
    sys.exit(app.exec_())
