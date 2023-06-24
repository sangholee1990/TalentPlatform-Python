# -*- coding: utf-8 -*-
import glob
import os
import platform
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QComboBox, QRadioButton, QHBoxLayout)
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

import zipfile
import os
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import pytz
# from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QBasicTimer
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QEventLoop, QTimer

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
    # 구글 API 정보
    'googleApiKey': ''
}

# 전역 설정
# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['axes.unicode_minus'] = False

# 메인 윈도우 클래스 정의
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.select_all_status = True
        self.select_all_status2 = True
        self.isStopProc = False

    def initUI(self):
        # 윈도우 타이틀 및 아이콘 설정
        # self.setWindowTitle('PyQt5 원도우 GUI 기반 지오코딩 프로그램')
        self.setWindowTitle('RIA-Geocoding')
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
        # self.column_combo.addItem('선택')

        grid.addWidget(self.search_label, 0, 0, alignment=Qt.AlignCenter)
        grid.addWidget(self.search_edit, 0, 1)
        grid.addWidget(self.column_label, 0, 2, alignment=Qt.AlignCenter)
        grid.addWidget(self.column_combo, 0, 3)

        # 대상 파일 영역 위젯 생성 및 배치
        self.select_all_button = QPushButton('전체 선택')
        self.select_all_button.clicked.connect(lambda: self.selectFileCheck(self.file_table, 'select_all_status'))
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(['선택', '파일명'])
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        self.upload_button = QPushButton('파일 업로드')
        self.upload_button.clicked.connect(self.upload_files)
        self.convert_button = QPushButton('위경도 변환')
        self.convert_button.clicked.connect(self.convert_files)
        self.delete_button = QPushButton('삭제')
        self.delete_button.clicked.connect(lambda: self.delete_files(self.file_table))

        grid.addWidget(self.select_all_button, 1, 0)
        grid.addWidget(self.upload_button, 1, 1)
        grid.addWidget(self.convert_button, 1, 2)
        grid.addWidget(self.delete_button, 1, 3)
        grid.addWidget(self.file_table, 2, 0, 1, 4)

        # 변환 파일 영역 위젯 생성 및 배치
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(['선택', '파일명'])
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        self.select_all_button2 = QPushButton('전체 선택')
        self.select_all_button2.clicked.connect(lambda: self.selectFileCheck(self.result_table, 'select_all_status2'))
        self.download_button = QPushButton('다운로드')
        self.download_button.clicked.connect(self.download_files)
        self.delete_button2 = QPushButton('삭제')
        self.delete_button2.clicked.connect(lambda: self.delete_files(self.result_table))
        self.pbar = QProgressBar(self)
        self.setLayout(grid)

        grid.addWidget(self.select_all_button2, 3, 0)
        grid.addWidget(self.download_button, 3, 1)
        grid.addWidget(self.delete_button2, 3, 2)
        grid.addWidget(self.pbar, 3, 3)
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

    # [화면 GUI]에서 중앙 정렬 기능 
    def center(self):
        # 윈도우를 화면 가운데로 이동하는 메소드
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # [대상 목록]에서 [파일 업로드] 기능
    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, '파일 업로드', '', 'CSV files (*.csv);;Excel files (*.xlsx)')

        itemList = [self.column_combo.itemText(i) for i in range(self.column_combo.count())]

        # 콤보박스를 초기화한다.
        if len(files) < 1 or len(itemList) < 1:
            self.column_combo.clear()

        # 중복되지 않는 컬럼명을 저장할 set
        columns_set = set()
        for file in files:
            # filename = file.split('/')[-1]
            if self.isFileDup(file, self.file_table):
                self.show_toast_message('[변환 목록] 파일 중복 발생')
                continue

            # [대상 목록]에서 행 추가
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 1, QTableWidgetItem(file))
            self.file_table.setCellWidget(row, 0, self.createCheckBox(True, self.file_table))

            # 파일 읽기
            df = self.read_file(file)

            if df is None or len(df) < 1:
                self.show_toast_message('[대상 목록] 파일 읽기 실패')
                continue

            # 데이터프레임의 컬럼명을 set에 추가한다.
            if df is not None and len(df.columns) > 0:
                columns_set.update(df.columns)

        # set의 모든 요소를 콤보박스에 추가한다.
        self.column_combo.addItems(list(columns_set))

    # [대상/변환 목록]에서 파일 중복 검사
    def isFileDup(self, filename, table):
        for i in range(table.rowCount()):
            if filename == table.item(i, 1).text():
                return True
        return False

    # [대상 목록]에서 위경도 변환 및 [변환 목록] 추가
    def convert_files(self):
        checked_files = [self.file_table.cellWidget(i, 0).layout().itemAt(0).widget().isChecked() for i in range(self.file_table.rowCount()) if isinstance(self.file_table.cellWidget(i, 0).layout().itemAt(0).widget(), QCheckBox)]

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

        try:
            gmaps = googlemaps.Client(key=key)
            # self.show_toast_message('구글 API키 인증 완료')
        except Exception as e:
            self.show_toast_message('구글 API키를 인증해 주세요.')
            return False

        selected_column = self.column_combo.currentText()

        for i in range(self.file_table.rowCount()):
            widget = self.file_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            filename = self.file_table.item(i, 1).text()

            filePath = os.path.dirname(filename)
            fileNameNoExt = os.path.basename(filename).split('.')[0]

            saveFile = f'{filePath}/{fileNameNoExt}_위경도 변환.csv'
            if self.isFileDup(saveFile, self.result_table):
                self.show_toast_message('[대상 목록] 파일 중복 발생')
                continue

            df = self.read_file(filename)

            if df is None or len(df) < 1:
                self.show_toast_message('[대상 목록] 파일 읽기 실패')
                continue

            # 구글 위경도 변환
            addrList = set(df[selected_column])

            # 진행바 생성
            maxCnt = len(addrList)
            self.pbar.setMaximum(maxCnt)
            self.pbar.setValue(0)

            matData = pd.DataFrame()
            for j, addrInfo in enumerate(addrList):

                if self.isStopProc:
                    break

                if j % 10 == 0:
                    # print(f'[CHECK] j : {j}')
                    self.pbar.setValue(j)
                    self.pbar.setFormat("{:.2f}%".format((j + 1) / maxCnt * 100))
                    QApplication.processEvents()

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

            self.isStopProc = False
            self.pbar.setValue(maxCnt)
            self.pbar.setFormat("{:.2f}%".format((j + 1) / maxCnt * 100))
            QApplication.processEvents()

            # 파일 저장
            df.to_csv(saveFile, index=False)

            # [변환 목록]에 행 추가
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 1, QTableWidgetItem(saveFile))
            self.result_table.setCellWidget(row, 0, self.createCheckBox(True, self.result_table))

    # [대상/변환 목록]에서 [삭제] 기능
    def delete_files(self, table):
        rows = []
        for i in range(table.rowCount()):
            widget = table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            rows.append(i)

        if len(rows) < 1:
            self.show_toast_message('삭제 파일을 선택해 주세요.')
            return False

        rows.reverse()

        for row in rows:
            table.removeRow(row)

        if len(rows) > 0:
            self.show_toast_message('삭제')

    # [대상/변환 목록]에서 [전체 선택] 기능
    def selectFileCheck(self, table, status_attribute):
        status = getattr(self, status_attribute)
        setattr(self, status_attribute, not status)
        for i in range(table.rowCount()):
            widget = table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()
            check.setChecked(getattr(self, status_attribute))

    # [변환 목록]에서 [다운로드] 기능
    def download_files(self):
        rows = []
        for i in range(self.result_table.rowCount()):
            widget = self.result_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()
            if not check.isChecked(): continue
            rows.append(i)

        if len(rows) < 1:
            self.show_toast_message('다운로드 파일을 선택해 주세요.')
            return False

        for row in rows:
            filename = self.result_table.item(row, 1).text()
            filePathFirst = os.path.dirname(filename)
            break

        zipFile = f'{filePathFirst}/{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")}_다운로드.zip'
        with zipfile.ZipFile(zipFile, "w") as zipf:
            for row in rows:
                filename = self.result_table.item(row, 1).text()
                zipf.write(filename, arcname=os.path.basename(filename))

            if (os.path.exists(zipFile)):
                self.show_toast_message(f'다운로드 완료 : {zipFile}', 3000)
            else:
                self.show_toast_message("다운로드 실패")

    # 메시지 알림
    def show_toast_message(self, message, display=1000):
        toast = QLabel(message, self)
        toast.setStyleSheet("background-color:#333;color:#fff;padding:8px;border-radius:4px;")
        toast.setAlignment(Qt.AlignCenter)
        toast.setGeometry(10, 10, toast.sizeHint().width(), toast.sizeHint().height())
        toast.show()
        toast.raise_()
        QTimer.singleShot(display, toast.close)

    # [대상/변환 목록]에서 상태 활성화/비활성화
    def setCheckState(self, table):
        for i in range(table.rowCount()):
            widget = table.cellWidget(i, 0)
            checkbox = widget.layout().itemAt(0).widget()

            if checkbox.isChecked():
                self.show_toast_message('선택')
            else:
                self.show_toast_message('해제')

    # [대상/변환 목록]에서 체크 박스 생성
    def createCheckBox(self, checked, table):
        check = QCheckBox()
        check.setChecked(checked)
        check.stateChanged.connect(lambda: self.setCheckState(table))

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(check)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        return widget

    # [CSV 또는 XLSX] 파일 읽기 기능
    def read_file(self, filename):

        df = pd.DataFrame()
        extension = filename.split('.')[-1]

        if extension == 'csv':
            encList = ['EUC-KR', 'UTF-8', 'CP949']
            for enc in encList:
                try:
                    df = pd.read_csv(filename, encoding=enc)
                    return df
                except Exception as e:
                    continue

        elif extension == 'xlsx':
            df = pd.read_excel(filename)
            return df
        else:
            return df

    # [X 버튼] 선택
    def closeEvent(self, event):
        self.isStopProc = True
        event.accept()

if __name__ == '__main__':
    if (platform.system() == 'Windows'):
        pass
    else:
        os.environ['DISPLAY'] = 'localhost:10.0'
        display_value = os.environ.get('DISPLAY')

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    # mw = qtmodern.windows.ModernWindow(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())