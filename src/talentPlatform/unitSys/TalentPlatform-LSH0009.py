# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import sys
# from plotnine import *
# from plotnine.data import *
# from dfply import *
# import hydroeval
import dfply
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import glob
import pprint
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler
# from keras.layers import LSTM
# from keras.models import Sequential
#
# from keras.layers import Dense
# import keras.backend as K
# from keras.callbacks import EarlyStopping
from multiprocessing import Pool, Process
import traceback
import sys
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from dfply import filter_by, group_by, summarize, ungroup, arrange, n, X
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 초기 환경변수 정의
from src.talentPlatform.unitSysHelper.InitConfig import *


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 웹 크롤링 및 워드 클라우드 시각화

    # 제출할 내용 :
    # 파이썬 코드 파일
    # 단어 구름 시각화를위한 이미지 파일

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0009'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 파이썬 실행 시 전달인자 설정
            # pyhton3 *.py argv1 argv2 argv3 ...
            for i, key in enumerate(inParams):
                if globalVar['sysOs'] in 'Linux':
                    if i >= len(sys.argv[1:]): continue
                    if inParams[key] == None: continue
                    val = inParams[key] if sys.argv[i + 1] == None else sys.argv[i + 1]

                if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
                    if inParams[key] == None: continue
                    val = inParams[key]

                # self 변수에 할당
                # setattr(self, key, val)

                # 전역 변수에 할당
                globalVar[key] = val
                log.info("[CHECK] {} / val : {}".format(key, val))

                print(os.getcwd())

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar key / val : {} / {}".format(key, val))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        try:
            log.info('[START] {}'.format("exec"))

            # breakpoint()

            # python -c "import nltk; nltk.download('punkt')"
            # nltk.download('stopwords')

            # 1) https://edition.cnn.com/2020/06/02/world/nodosaur-fossil-stomach-contents-scn-trnd/index.html에서 기사 내용을 스크랩하십시오.
            html = urlopen(
                "https://edition.cnn.com/2020/06/02/world/nodosaur-fossil-stomach-contents-scn-trnd/index.html")
            # html = requests.get(url)
            soup = BeautifulSoup(html, 'html.parser')

            section = soup.select('section.zn-body-text')

            liGetText = []
            for i in section:
                getText = i.get_text()

                log.info("getText : {%s} : {%s}", len(getText), getText)

                # 단어 추출
                wordTokens = word_tokenize(getText)
                # 불용어
                stopWords = set(stopwords.words('english'))

                # log.info("wordTokens : {%s} : {%s}", len(wordTokens), wordTokens)
                # log.info("stopWords : {%s} : {%s}", len(stopWords), stopWords)

                # 2) 기사 내용을 사전 처리하여 불용어없이 단수 명사 목록을 얻습니다.
                for j in wordTokens:
                    if j not in stopWords:
                        liGetText.append(j)

            log.info("liGetText : {%s} : {%s}", len(liGetText), liGetText)

            data = pd.DataFrame({
                'type': liGetText
            })

            # 3) 빈도분포 및 워드 클라우드 시각화
            dataL1 = (
                (
                        data >>
                        filter_by(
                            X.type != '.'
                            , X.type != ','
                            , X.type != "'"
                            , X.type != "''"
                            , X.type != "``"
                            , X.type != "'s"
                        ) >>
                        group_by(X.type) >>
                        summarize(number=n(X.type)) >>
                        ungroup() >>
                        arrange(X.number, ascending=False)
                )
            )

            log.info("dataL1 : {%s} : {%s}", len(dataL1), dataL1)

            # 데이터 시각화를 위한 전처리
            objData = {}
            for i in dataL1.values:
                key = i[0]
                val = i[1]

                objData[key] = val

            log.info("objData : {%s} : {%s}", len(objData), objData)

            wordcloud = WordCloud(
                width=1000
                , height=1000
                , background_color="white"
            ).generate_from_frequencies(objData)

            saveImg = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '워드 클라우드.png')
            log.info('[CHECK] saveFile : {}'.format(saveImg))

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(saveImg, dpi=600, bbox_inches='tight')
            plt.show()

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

    # 수행 프로그램 (단일 코어, 다중 코어 멀티프레세싱)
    def runPython(self):
        try:
            log.info('[START] {}'.format("runPython"))

            DtaProcess.exec(self)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("runPython"))


if __name__ == '__main__':

    try:
        log.info('[START] {}'.format("main"))

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        # 입력 자료 : inpPath
        # 그림 자료 : figPath
        # 출력 자료 : outPath
        # 로그 자료 : logPath
        inParams = {
            # 'inpPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'figPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'outPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'logPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
        }

        log.info("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        log.error(traceback.format_exc())
        sys.exit(1)

    finally:
        log.info('[END] {}'.format("main"))