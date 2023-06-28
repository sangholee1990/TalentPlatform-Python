# -*- coding: utf-8 -*-

import traceback

from PIL import Image

# 초기 환경변수 정의
from src.talentPlatform.unitSys.helper.InitConfig import *
from src.talentPlatform.unitSys.helper.central_limit_theorem import CentralLimitTheorem


class DtaProcess(object):
    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0007'

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

                if globalVar['sysOs'] in 'Windows':
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

            fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], 'LSH0007_lena_gray.gif'))
            if (len(fileInfo1) < 1): raise Exception("[ERROR] fileInfo1 : {}".format("자료를 확인해주세요."))

            # 이미지 읽기
            image = Image.open(fileInfo1[0])

            arrVal2D = np.array(image)
            arrVal1D = arrVal2D.flatten()

            log.info("=================== 과제 1 ===================")
            log.info("모집단 크기 : {%s}", arrVal1D.size)
            log.info("모집단 평균 : {%s}", round(np.mean(arrVal1D), 2))
            log.info("모집단 분산 : {%s}", round(np.var(arrVal1D), 2))
            log.info("모집단 최대값 : {%s}", np.max(arrVal1D))
            log.info("모집단 최소값 : {%s}", np.min(arrVal1D))
            log.info("모집단 중앙값 : {%s}", np.median(arrVal1D))

            log.info("=================== 과제 2 ===================")
            plt.hist(arrVal1D)
            plt.show()

            log.info("=================== 과제 3 ===================")
            binList = [10, 100, 1000]

            for i in binList:
                plt.hist(arrVal1D, bins=i)
                plt.show()

            log.info("=================== 과제 4 ===================")
            callClt = CentralLimitTheorem(arrVal1D)
            sampleList = [5, 10, 20, 30, 50, 100]
            for sample in sampleList:
                callClt.run_sample(N=sample, plot=True, num_bins=None)

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

        #  # 파이썬 실행 시 전달인자 설정
        inParams = {
        }

        log.info("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)
        callDtaProcess.runPython()

    except Exception as e:
        log.error(traceback.format_exc())
        sys.exit(1)

    finally:
        log.info('[END] {}'.format("main"))