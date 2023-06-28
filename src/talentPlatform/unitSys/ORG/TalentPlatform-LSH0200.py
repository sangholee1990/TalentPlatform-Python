# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import glob
import sys
import logging
import platform
import sys
import traceback
import urllib
from datetime import datetime
from urllib import parse

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dfply import *
from plotnine.data import *
from sspipe import p, px
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =================================================
# 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


# =================================================
# 함수 정의
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

    globalVar = {
        "prjName": prjName
        , "sysOs": platform.system()
        , "contextPath": contextPath
        , "resPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , "cfgPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , "inpPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , "sysPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace("\\", "/")

    return globalVar

#  초기 전달인자 설정
def initArgument(globalVar, inParams):

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

    return globalVar

def reqHwpFileDown(inHwpFile, inCaSvephy):

    prefixUrl = 'https://gnews.gg.go.kr/Operator/reporter_room/notice/download.do?'

    reqHwpUrl = (
            '{}file={}&BS_CODE=s017&CA_SAVEPHY={}'.format(prefixUrl, inHwpFile, inCaSvephy)
            | p(parse.urlparse).query
            | p(parse.parse_qs)
            | p(parse.urlencode, doseq=True)
            | prefixUrl + px
    )

    saveHwpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, inHwpFile)

    # 디렉터리 없을 시 생성
    if not os.path.exists(os.path.dirname(saveHwpFile)):
        os.makedirs(os.path.dirname(saveHwpFile))

    # 파일 존재 유무 판단
    isFile = os.path.exists(saveHwpFile)

    # if isFile: return Pa

    res = urllib.request.urlopen(reqHwpUrl)
    resCode = res.getcode()
    resSize = int(res.headers['content-length'])

    if resCode != 200:
        return False

    if resSize < 82:
        return False

    with open(saveHwpFile, mode="wb") as f:
        f.write(res.read())

    log.info('[CHECK] saveHwpFile : {} / {} / {}'.format(inCaSvephy, isFile, saveHwpFile))

    return True

# 상관계수 행렬 시각화
def makeCorrPlot(data, saveImg):
    corr = data.corr(method='pearson')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, square=True, annot=False, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()

class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # LSH0200. Python을 이용한 R 코드 변환 및 회귀분석

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0200'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

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

            breakpoint()

            # 파일 조회
            contentInfo = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0200_(공유)임직원+데이터v2.xlsx'))

            # 파일 없을 시 에러 발생
            if (len(contentInfo) < 1): raise Exception("[ERROR] contentInfo : {}".format("자료를 확인해주세요."))

            # 엑셀 파일 읽기
            data = pd.read_excel(contentInfo[0], sheet_name='Raw_data')

            # 상관계수 행렬에 대한 이미지 저장 파일명
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '임직원 데이터 상관계수 행렬')
            # 상관계수 행렬 시각화
            makeCorrPlot(data, saveImg)

            # 독립 변수 : place_level, inter_count, ws_count
            # 종속 변수 : commit_job, commit_org, commit_rel
            colList = ['commit_job', 'commit_org', 'commit_rel']

            colInfo = 'commit_job'
            # 종속 변수에 따른 반복문 수행
            for i, colInfo in enumerate(colList):
                # 동적 회귀 모형식
                formula = '{} ~ place_level'.format(colInfo)
                formula = '{} ~ place_level + inter_count'.format(colInfo)
                formula = '{} ~ place_level + inter_count + place_level * inter_count'.format(colInfo)
                formula = '{} ~ place_level + inter_count + ws_count'.format(colInfo)
                formula = '{} ~ place_level + inter_count + ws_count + place_level*inter_count + place_level*ws_count + inter_count*ws_count'.format(colInfo)

                # ***************************************************
                # 모든 변수에 대한 다중선형회귀모형
                # ***************************************************
                # smModel = sm.OLS.from_formula(formula, data)
                # result = smModel.fit()
                # result.summary()

                # 베타 계수 계산
                # betaCoef = np.array(result.params)[1:] * (smModel.exog.std(axis=0)[1:] / smModel.endog.std(axis=0))
                # print(betaCoef)

                # ***************************************************
                # 표준화 다중선형회귀모형
                # ***************************************************
                # 데이터 표준화
                dataL1 = data.apply(stats.zscore)

                # 회귀모델 수행
                smfModel = smf.ols(formula, data=dataL1)

                # 회귀모델에 대한 예측 결과
                resultL1 = smfModel.fit()

                # 결과 요약
                resultL1.summary()

                print(resultL1.summary())

                # ***************************************************
                # 다중공선성 계산
                # ***************************************************
                dataL2 = pd.DataFrame()

                # 회귀모형에서 독립변수에 대한 반복문 수행
                for j, colInfo in enumerate(smfModel.exog_names):
                    # 절편 제외
                    if colInfo == 'Intercept': continue

                    # 독립변수 및 다중공선성 계산
                    dict = {
                        'col': [colInfo]
                        , 'vif': [variance_inflation_factor(smfModel.exog, j)]
                    }

                    # 앞서 데이터프레임에서 행 단위로 추가
                    dataL2 = dataL2.append(pd.DataFrame.from_dict(dict))

                print(dataL2)
            # ***************************************************
            # 결과 분석
            # ***************************************************
            # 독립변수 : 근무지급지, 직무인터뷰, 귀임자 workshop
            # 종속변수 : 직무만족도
            # 귀임자 workshop (0.2088), 직무인터뷰 (0.3718), 근무지급지 (0.4322)이 높을수록 직무만족도가 높음
            # 특히 근무지급지일 때 가장 높은 직무만족도를 보임

            # 독립변수 : 근무지급지, 직무인터뷰, 귀임자 workshop
            # 종속변수 : 조직만족도
            # 직무인터뷰 (0.1829), 근무지급지 (0.3767), 귀임자 workshop (0.4529)이 높을수록 조직만족도가 높음
            # 특히 귀임자 workshop일 때 가장 높은 조직만족도를 보임

            # 독립변수 : 근무지급지, 직무인터뷰, 귀임자 workshop
            # 종속변수 : 관계만족도
            # 직무인터뷰 (0.1466), 근무지급지 (0.2209), 귀임자 workshop (0.2964)이 높을수록 관계만족도가 높음
            # 특히 귀임자 workshop일 때 가장 높은 관계만족도를 보임

            # ***************************************************
            # QA 답변
            # ***************************************************
            # (1) 근무지 급지 독립변수 -----> 종속 변수(직무만족도, 조직만족도, 관계 만족도)
            # 근무지 급지가 높을 수록 좋은 환경이고, 높을수록 종속변수도 정의 관계로 증가
            # A. 앞서 결과 분석에서와 같이 독립변수 (근무지급지, 직무인터뷰, 귀임자 workshop)에 따라 종속변수 (직무/조직/관계 만족) 증가

            # (2) 각 변수간의 상관관계도 높음 확인
            # A. 그림. 상관계수 행렬도를 참조

            # (3) 조절변수로 직무 인터뷰 진행횟수 적용,
            # 직무 인터뷰를 진행하면, 횟수가 늘어날수록 종속변수 증가(단, 횟수와 급지가 높은 것을 상승률 저하)
            # A. 직무인터뷰가 증가할수록 종속변수 (직무만족도, 조직만족도, 관계 만족도)를 모두 증가
            # A. 그에 따른 상대적인 비율의 경우 직무 만족도일 때 가장 높음

            # (4) 조절변수로 WS 참여 횟수 적용
            # 임직원 대상으로 WS을 진행하면 조직/관계 만족도 상승 효과
            # A. 귀임자 workshop (WS)가 증가할수록 종속변수 (직무만족도, 조직만족도, 관계 만족도)를 모두 증가
            # A. 그에 따른 상대적인 비율의 경우 조직 만족도, 관계 만족도일 때 가장 높음

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

    # 수행 프로그램 (단일 코어, 다중 코어 멀티프로세싱)
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
        print('[START] {}'.format("main"))

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

        print("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
