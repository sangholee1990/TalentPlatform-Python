# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import requests
import json

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램 :부 프로그램을 호출
# 4. 부 프로그램 : 자료 처리를 위한 클래스로서 내부 함수 (초기 변수, 비즈니스 로직, 수행 프로그램 설정)
# 4.1. 환경 변수 설정 (로그 설정) : 로그 기록을 위한 설정 정보 읽기
# 4.2. 환경 변수 설정 (초기 변수) : 입력 경로 (inpPath) 및 출력 경로 (outPath) 등을 설정
# 4.3. 초기 변수 (Argument, Option) 설정 : 파이썬 실행 시 전달인자 설정 (pyhton3 *.py argv1 argv2 argv3 ...)
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리 또는 비즈니스 로직 구현

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
    )

    if not os.path.exists(os.path.dirname(saveLogFile)):
        os.makedirs(os.path.dirname(saveLogFile))

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile)

    # logger instance에 format 설정
    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    # logger instance에 handler 설정
    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    # logger instance로 log 기록
    log.setLevel(level=logging.INFO)

    return log


#  초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    # 환경 변수 (local, 그 외)에 따라 전역 변수 (입력 자료, 출력 자료 등)를 동적으로 설정
    # 즉 local의 경우 현재 작업 경로 (contextPath)를 기준으로 설정
    # 그 외의 경우 contextPath/resources/input/prjName와 같은 동적으로 구성
    globalVar = {
        'prjName': prjName
        , 'sysOs': platform.system()
        , 'contextPath': contextPath
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    # 원도우 또는 맥 환경
    if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
        inParInfo = inParams

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

    log.info("[CHECK] inParInfo : {}".format(inParInfo))

    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 3차원 극한가뭄 빈도수 및 상자그림

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0441'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '1990-01-01'
                    , 'endDate': '2022-01-01'

                    # 목록
                    , 'keyList': ['ACCESS-CM2']

                    # 극한 가뭄값
                    , 'extDrgVal': -2
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'dateList': {
                        'all': ('2000-01-01', '2100-12-31')
                        , 'case': ('2031-01-01', '2065-12-31')
                        , 'case2': ('2066-01-01', '2100-12-31')
                    }

                    # 목록
                    , 'keyList': ['ACCESS-CM2']

                    # 가뭄 목록
                    , 'drgCondList': {
                        'EW': (2.0, 4.0)
                        , 'VW': (1.50, 1.99)
                        , 'MW': (1.00, 1.49)
                        , 'NN': (-0.99, 0.99)
                        , 'MD': (-1.00, -1.49)
                        , 'SD': (-1.50, -1.99)
                        , 'ED': (-2.00, -4.00)
                    }

                    # 극한 가뭄값
                    , 'extDrgVal': -2
                }


                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            import os

            os.environ['DISPLAY'] = 'localhost:10.0'

            display_value = os.environ.get('DISPLAY')
            print(display_value)

            import sys
            from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                                         QLineEdit, QFileDialog, QLabel, QRadioButton, QHBoxLayout)

            from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QComboBox)


            # from geopy.geocoders import GoogleV3
            from geopy.geocoders import Nominatim
            import pandas as pd
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

                    # API 키 입력창
                    self.api_key_entry.setPlaceholderText("API 키를 입력하세요.")
                    upload_button = QPushButton('파일 업로드')
                    upload_button.clicked.connect(self.upload_file)

                    # 변환 버튼
                    convert_button = QPushButton('변환')
                    convert_button.clicked.connect(self.geocode)

                    # 레이아웃 설정
                    layout = QVBoxLayout()
                    layout.addWidget(QLabel('API Key:'))
                    layout.addWidget(self.api_key_entry)
                    layout.addWidget(upload_button)
                    layout.addWidget(self.file_entry)
                    layout.addWidget(QLabel('주소 컬럼 선택:'))
                    layout.addWidget(self.column_combo)
                    layout.addWidget(convert_button)

                    self.setLayout(layout)
                    self.show()

                def upload_file(self):
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
                app = QApplication(sys.argv)
                ex = MyApp()
                sys.exit(app.exec_())



            # GUI 생성
            window = tk.Tk()
            window.title("지오코딩 변환기")

            # API 키 입력창
            api_label = tk.Label(window, text="Google API 키:")
            api_label.pack()
            api_entry = tk.Entry(window)
            api_entry.pack()

            # 파일 경로 출력창
            file_label = tk.Label(window, text="파일 경로:")
            file_label.pack()
            file_entry = tk.Entry(window)
            file_entry.pack()

            # 주소를 위경도로 변환하는 함수
            def geocode():
                # API 키 가져오기
                api_key = api_entry.get()

                # 파일 업로드 다이얼로그
                file_path = filedialog.askopenfilename()
                file_entry.delete(0, tk.END)
                file_entry.insert(0, file_path)

                # 파일 불러오기
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:  # xlsx 파일로 가정
                    data = pd.read_excel(file_path, engine='openpyxl')

                # 위경도 변환
                geocoded = []
                for address in data['address']:  # 'address'는 주소를 포함하는 열 이름
                    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
                    response = requests.get(url)
                    json_response = json.loads(response.text)
                    if json_response['status'] == 'OK':
                        location = json_response['results'][0]['geometry']['location']
                        geocoded.append([location['lat'], location['lng']])
                    else:
                        geocoded.append([None, None])

                # 변환된 결과를 새 열로 추가
                data['latitude'], data['longitude'] = zip(*geocoded)

                # 파일 저장 다이얼로그
                save_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
                if save_path.endswith('.csv'):
                    data.to_csv(save_path, index=False)
                else:  # xlsx 파일로 가정
                    data.to_excel(save_path, index=False, engine='openpyxl')

            # 변환 버튼
            convert_button = tk.Button(window, text="변환", command=geocode)
            convert_button.pack()

            window.mainloop()

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = {}

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
