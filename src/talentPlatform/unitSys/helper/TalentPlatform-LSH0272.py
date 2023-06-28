# -*- coding: utf-8 -*-
import glob
# import seaborn as sns
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
import tifffile as tiff
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Activation, concatenate

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

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace('\\', '/')

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    for i, key in enumerate(inParams):
        # 리눅스 환경
        if globalVar['sysOs'] in 'Linux':
            if i >= len(sys.argv[1:]): continue
            if inParams[key] is None: continue
            val = inParams[key] if sys.argv[i + 1] is None else sys.argv[i + 1]

        # 원도우 또는 맥 환경
        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] is None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

def Encoder(x, filters=64):
    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pool = MaxPool2D(pool_size=(2, 2))(x)

    return x, pool


def Decoder(x, _c=None, filters=64):
    if _c != None:
        x = concatenate([x, _c], axis=-1)

        x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        pool = MaxPool2D(pool_size=(2, 2))(x)

        return x


def Outblock(x, _c=None, filters=64):
    if _c != None:
        x = concatenate([x, _c], axis=-1)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 위성영상 토지피복분류

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정


    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0272'

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
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar[{}] {}".format(key, val))

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

            # **********************************************************************************************************
            #  옵션 설정
            # **********************************************************************************************************
            N = 512

            # image size
            M = 512

            # band number
            B = 4

            # **********************************************************************************************************
            #  FGT 목록 파일 읽기, 찾기
            # **********************************************************************************************************
            # FGT 파일
            inpFileFgt = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Sentinel_Segmentation/fgt_list')

            # # FGT 파일 찾기
            fileInfoFgt = glob.glob(inpFileFgt)

            # FGT 파일 없을 경우 예외 처리
            if fileInfoFgt is None or len(fileInfoFgt) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileInfoFgt, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileInfoFgt, '입력 자료를 확인해주세요.'))

            with open(fileInfoFgt[0]) as myfile:
                annot_q1 = myfile.readlines()


            # **********************************************************************************************************
            #  IMAGE 목록 파일 읽기, 찾기
            # **********************************************************************************************************
            # IMAGE 파일
            inpFileImg = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Sentinel_Segmentation/img_list')

            # IMAGE 파일 찾기
            fileInfoImg = glob.glob(inpFileImg)

            # IMAGE 파일 없을 경우 예외 처리
            if fileInfoImg is None or len(fileInfoImg) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileInfoImg, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileInfoImg, '입력 자료를 확인해주세요.'))

            with open(fileInfoImg[0]) as myfile:
                img_q1 = myfile.readlines()

            # **********************************************************************************************************
            #  자료 전처리
            # **********************************************************************************************************
            # 초기값 설정
            dataFrame_orig = np.zeros((len(img_q1), N, M, B), 'float32')
            labelFrame_orig = np.zeros((len(img_q1), N, M), 'float32')

            # 데이터 및 라벨링 용도의 이미지 영상 읽기
            for i in range(len(img_q1)):
                inpFileFgt = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Sentinel_Segmentation/FGT_TIF/')
                inpFileImg = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Sentinel_Segmentation/IMAGE/')

                labelFrame_orig[i] = tiff.imread(inpFileFgt + annot_q1[i][:23])
                dataFrame_orig[i] = tiff.imread(inpFileImg + img_q1[i][:19])

            # **********************************************************************************************************
            #  라벨링에 대한 의미 부여
            # **********************************************************************************************************
            # 건물
            labelFrame_orig[labelFrame_orig == 10] = 4

            # 도로
            labelFrame_orig[labelFrame_orig == 30] = 1

            # 논
            labelFrame_orig[labelFrame_orig == 50] = 2

            # 밭
            labelFrame_orig[labelFrame_orig == 60] = 2

            # 숲
            labelFrame_orig[labelFrame_orig == 70] = 3

            # background class
            labelFrame_orig[labelFrame_orig == 100] = 0

            # **********************************************************************************************************
            #  훈련 및 테스트 데이터셋 8:2 분류
            # **********************************************************************************************************
            train_ratio = 0.8

            # 독립변수에 대한 정규화 수행
            for i in range(dataFrame_orig.shape[0]):
                for b in range(4):
                    _max = np.max(dataFrame_orig[i, :, :, b])
                    _min = np.min(dataFrame_orig[i, :, :, b])

                    dataFrame_orig[i, :, :, b] = (dataFrame_orig[i, :, :, b] - _min) / (_max - _min)

            # 훈련 및 테스트 데이터셋 인덱스 설정
            nTrain = int(labelFrame_orig.shape[0] * train_ratio)
            nTest = labelFrame_orig.shape[0] - nTrain

            # 훈련 데이터셋에서 독립변수 및 종속변수 설정
            x_train = dataFrame_orig[:nTrain].astype('float32')
            t_train = labelFrame_orig[:nTrain].astype('float32')

            # 테스트 데이터셋에서 독립변수 및 종속변수 설정
            x_test = dataFrame_orig[nTrain:].astype('float32')
            t_test = labelFrame_orig[nTrain:].astype('float32')

            # 데이터 열 기준으로 합치기 (data stacking)
            x_train = np.vstack([x_train, x_train])
            t_train = np.vstack([t_train, t_train])
            x_test = np.vstack([x_test, x_test])
            t_test = np.vstack([t_test, t_test])
            N_train = nTrain * 2
            N_test = nTest * 2

            # 랜덤 데이터를 위한 무작위 정렬 (Shuffling)
            vec_train = np.arange(N_train)
            vec_test = np.arange(N_test)

            np.random.shuffle(vec_train)
            np.random.shuffle(vec_test)

            x_train = x_train[vec_train]
            t_train = t_train[vec_train]

            x_test = x_test[vec_test]
            t_test = t_test[vec_test]

            # 이미지 영상 좌우 대칭 (flipping)
            nn = N_train//3
            x_train[:nn] = x_train[:nn, ::-1, :, :]
            t_train[:nn] = t_train[:nn, ::-1, :]

            x_train[nn:2*nn] = x_train[nn:2*nn, :, ::-1, :]
            t_train[nn:2*nn] = t_train[nn:2*nn, :, ::-1]

            nn = N_test//3
            x_train[:nn] = x_train[:nn, ::-1, :, :]
            t_train[:nn] = t_train[:nn, ::-1, :]

            x_train[nn:2 * nn] = x_train[nn:2 * nn, :, ::-1, :]
            t_train[nn:2 * nn] = t_train[nn:2 * nn, :, ::-1]

            # 랜덤 데이터를 위한 무작위 정렬 (Shuffling)
            vec_train = np.arange(N_train)
            vec_test = np.arange(N_test)
            np.random.shuffle(vec_train)
            np.random.shuffle(vec_test)
            x_train = x_train[vec_train]
            t_train = t_train[vec_train]
            x_test = x_test[vec_test]
            t_test = t_test[vec_test]

            # 훈련 데이터셋을 이용하여 이미지 영상 회전 (Rotating)
            nn = N_train//4
            for i in range(3):
                x_train[(i+1)*nn : (i+2)*nn] = np.rot90(x_train[(i)*nn : (i+1)*nn], axes=(1, 2))
                t_train[(i+1)*nn : (i+2)*nn] = np.rot90(t_train[(i)*nn : (i+1)*nn], axes=(1, 2))

            # 테스트 데이터셋을 이용하여 이미지 영상 회전 (Rotating)
            nn = N_test//4
            for i in range(3):
                x_test[(i + 1) * nn: (i + 2) * nn] = np.rot90(x_test[(i) * nn: (i + 1) * nn], axes=(1, 2))
                t_test[(i + 1) * nn: (i + 2) * nn] = np.rot90(t_test[(i) * nn: (i + 1) * nn], axes=(1, 2))

            # 랜덤 데이터를 위한 무작위 정렬 (Shuffling)
            np.random.shuffle(vec_train)
            np.random.shuffle(vec_test)
            x_train = x_train[vec_train]
            t_train = t_train[vec_train]
            x_test = x_test[vec_test]
            t_test = t_test[vec_test]

            # 랜덤 데이터를 위한 무작위 정렬 (Shuffling)
            print(f'input data size = {x_train.shape}')
            print(f'input data size = {t_train.shape}')

            # 훈련 데이터셋을 이용한 데이터 및 라벨링 결과 (상단 좌/우: 인덱스 1번, 하단 좌/우: 인덱스 2번)를 시각화하였다.
            # 즉 상단 좌측의 경우 RGB 영상으로 나타내었고 상단 우측에서는 6개 지표면 타입에 대해서 분류할 결과이다.
            # 하단 좌/우측도 앞선 설명한 바와 같이 유사한 그림으로서 분류 결과가 잘 일치함을 보인다.
            plt.figure(figsize=(15, 15))

            plt.subplot(2, 2, 1)
            plt.imshow(x_train[1, :, :, :])

            plt.subplot(2, 2, 2)
            plt.imshow(t_train[1, :, :])

            plt.subplot(2, 2, 3)
            plt.imshow(x_train[15, :, :, :])

            plt.subplot(2, 2, 4)
            plt.imshow(t_train[15, :, :])

            mainTitle = 'Landuse Classification'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            log.info('[CHECK] saveImg : {}'.format(saveImg))

            plt.savefig(saveImg, dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()


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
        inParams = { }

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
