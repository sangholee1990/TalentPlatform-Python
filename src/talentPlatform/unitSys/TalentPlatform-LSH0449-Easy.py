# -*- coding: utf-8 -*-
import glob
import os
import sys
import platform
import pandas as pd
import pandas as pd
# import matplotlib.pyplot as plt
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
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
import geopandas as gpd
import re
import shutil

# ============================================
# 요구사항
# ============================================
# Python을 이용한 원도우 GUI 기반 SHP 처리 프로그램 (SHP 기본 및 부가 자료 통합화)

# ============================================
# 보조
# ============================================

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0449'

# 옵션 설정
sysOpt = {
}

if platform.system() == 'Windows':
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

        self.initUI()
        self.select_all_status = True
        self.select_all_status2 = True
        self.isStopProc = False

    def initUI(self):
        # 윈도우 타이틀 및 아이콘 설정
        # self.setWindowTitle('PyQt5 원도우 GUI 기반 지오코딩 프로그램')
        self.setWindowTitle('Python을 이용한 원도우 GUI 기반 SHP 처리 프로그램')
        self.setWindowIcon(QIcon('icon.png'))

        # 그리드 레이아웃 생성
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

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
        self.convert_button = QPushButton('SHP 통합')
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
        # self.resize(1000, 800)
        self.resize(1200, 800)
        # self.resize(1500, 800)
        # self.center()

        # 윈도우 보이기
        self.show()

    # [화면 GUI]에서 중앙 정렬 기능
    # def center(self):
    #     # 윈도우를 화면 가운데로 이동하는 메소드
    #     qr = self.frameGeometry()
    #     cp = QDesktopWidget().availableGeometry().center()
    #     qr.moveCenter(cp)
    #     self.move(qr.topLeft())

    # [대상 목록]에서 [파일 업로드] 기능
    def upload_files(self):
        # files, _ = QFileDialog.getOpenFileNames(self, '파일 업로드', '', 'SHP files (*.shp);;Excel files (*.xlsx)')
        # files, _ = QFileDialog.getOpenFileNames(self, '파일 업로드', '', 'SHP files (*.shp)')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath = QFileDialog.getExistingDirectory(self, "폴더 선택", "", options=options)

        # 임시 폴더/파일 삭제
        tmpPath = os.path.join(filePath, '임시')
        if os.path.exists(tmpPath):
            shutil.rmtree(tmpPath, ignore_errors=True)

        # 압축 해제
        zipFileList = glob.glob(os.path.join(filePath, '*.zip'))

        for fileInfo in zipFileList:
            zipFileInfo = os.path.join(os.path.dirname(fileInfo), '임시', os.path.basename(fileInfo))
            os.makedirs(os.path.dirname(zipFileInfo), exist_ok=True)

            with zipfile.ZipFile(fileInfo, 'r') as zip_ref:
                zip_ref.extractall(zipFileInfo.split('.')[0])

        if zipFileList is None or len(zipFileList) < 1:
            self.show_toast_message(f'[오류] {filePath} 폴더에 zip 목록이 없습니다.', 3000)
            return

        # 파일 조회
        files = glob.glob(os.path.join(filePath, '임시/*/*.shp'), recursive=True)

        # print(f'[CHECK] files : {files}')

        if files is None or len(files) < 1:
            self.show_toast_message(f'[오류] {filePath} 폴더에 shp 목록이 없습니다.', 3000)
            return

        for file in files:
            if self.isFileDup(file, self.file_table):
                self.show_toast_message(f'[경고] 파일 중복 발생')
                continue

            # [대상 목록]에서 행 추가
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            self.file_table.setItem(row, 1, QTableWidgetItem(file))
            self.file_table.setCellWidget(row, 0, self.createCheckBox(True, self.file_table))

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
            self.show_toast_message(f'[오류] 대상 파일을 선택해 주세요.')
            return False

        keyList = []
        for i in range(self.file_table.rowCount()):
            widget = self.file_table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            filename = self.file_table.item(i, 1).text()
            filePath = os.path.dirname(os.path.dirname(filename))

            fileSep = re.escape(os.sep)
            key = re.search(r'100M_(.*?){}'.format(fileSep), filename).group(1)
            keyList.append(key)
        # print(f'[CHECK] keyList : {set(keyList)}')

        # 진행바 생성
        maxCnt = len(set(keyList))
        self.pbar.setMaximum(maxCnt)
        self.pbar.setValue(0)
        for j, keyInfo in enumerate(set(keyList)):

            if self.isStopProc:
                break

            self.pbar.setValue(j)
            self.pbar.setFormat("{:.2f}%".format((j + 1) / maxCnt * 100))
            QApplication.processEvents()

            saveFile = os.path.join(filePath, f'{keyInfo}_통합.shp')
            if self.isFileDup(saveFile, self.result_table):
                self.show_toast_message(f'[오류] 대상 파일 중복 발생')
                continue

            shpData = gpd.GeoDataFrame()
            for i in range(self.file_table.rowCount()):
                widget = self.file_table.cellWidget(i, 0)
                check = widget.layout().itemAt(0).widget()

                if not check.isChecked(): continue
                filename = self.file_table.item(i, 1).text()
                if (not re.search(keyInfo, filename)): continue

                # print(f'[CHECK] filename : {filename}')

                gdf = gpd.read_file(filename, encoding='UTF-8')
                # gdf = gpd.read_file(filename, encoding='CP949')
                # gdf = gpd.read_file(filename, encoding='EUC-KR')

                type = re.search(r'인구정보-(.*?)[ 인]', filename).group(1)
                gdf = gdf.rename({'val': str(type)}, axis='columns').drop('lbl', axis='columns')

                # SHP 통합
                if len(shpData) < 1:
                    shpData = gdf
                else:
                    shpData = shpData.merge(gdf, on=['gid', 'geometry'], how='left')

            # print(f'[CHECK] shpData : {shpData}')

            # 파일 저장
            if len(shpData) > 0:
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # shpData.to_file(saveFile, encoding='UTF-8')
                # shpData.to_file(saveFile, encoding='EUC-KR')
                shpData.to_file(saveFile, encoding='CP949')

                # [변환 목록]에 행 추가
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row, 1, QTableWidgetItem(saveFile))
                self.result_table.setCellWidget(row, 0, self.createCheckBox(True, self.result_table))

        self.isStopProc = False
        self.pbar.setValue(maxCnt)
        self.pbar.setFormat("{:.2f}%".format((j + 1) / maxCnt * 100))
        QApplication.processEvents()

    # [대상/변환 목록]에서 [삭제] 기능
    def delete_files(self, table):
        rows = []
        for i in range(table.rowCount()):
            widget = table.cellWidget(i, 0)
            check = widget.layout().itemAt(0).widget()

            if not check.isChecked(): continue
            rows.append(i)

        if len(rows) < 1:
            self.show_toast_message(f'[오류] 삭제 파일을 선택해 주세요.')
            return False

        rows.reverse()

        for row in rows:
            table.removeRow(row)

        if len(rows) > 0:
            self.show_toast_message(f'[완료] 삭제 완료했습니다.')

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
            self.show_toast_message(f'[오류] 다운로드 파일을 선택해 주세요.')
            return False

        for row in rows:
            filename = self.result_table.item(row, 1).text()
            filePath = os.path.dirname(filename)
            fileNameNoExt = os.path.basename(filename).split('.')[0]
            fileList = glob.glob(f'{filePath}/{fileNameNoExt}.*')

            zipFile = f'{filePath}/{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")}_{fileNameNoExt}_다운로드.zip'
            with zipfile.ZipFile(zipFile, "w") as zipf:
                for fileInfo in fileList:
                    zipf.write(fileInfo, arcname=os.path.basename(fileInfo))

            if (os.path.exists(zipFile)):
                self.show_toast_message(f'[완료] 다운로드 완료 : {zipFile}', 3000)
            else:
                self.show_toast_message(f"[실패] 다운로드 실패")

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
    # def read_file(self, filename):
    #
    #     df = pd.DataFrame()
    #     extension = filename.split('.')[-1]
    #
    #     if extension == 'csv':
    #         encList = ['EUC-KR', 'UTF-8', 'CP949']
    #         for enc in encList:
    #             try:
    #                 df = pd.read_csv(filename, encoding=enc)
    #                 return df
    #             except Exception as e:
    #                 continue
    #
    #     elif extension == 'xlsx':
    #         df = pd.read_excel(filename)
    #         return df
    #     else:
    #         return df

    # [X 버튼] 선택
    def closeEvent(self, event):
        self.isStopProc = True
        event.accept()

if __name__ == '__main__':
    if platform.system() == 'Windows':
        pass
    else:
        # pass
        os.environ['DISPLAY'] = 'localhost:10.0'
        display_value = os.environ.get('DISPLAY')

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())