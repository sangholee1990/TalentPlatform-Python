
# 라이브러리 읽기
import logging as log
import os
import sys
import math
import sys
import traceback
import warnings
import numpy as np

class DtaProcess():

    # 로그 설정
    log.basicConfig(stream=sys.stdout, level=log.INFO,
                    format="%(asctime)s [%(filename)s > %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
    warnings.filterwarnings("ignore")

    # ================================================================================================
    # 초기 파라미터 설정
    # ================================================================================================
    def __init__(self):
        log.info("[START] __init__ : {}".format("init"))

        try:
            if len(sys.argv) < 0:
                # print(self.USAGE)
                log.error("매개변수를 확인해주세요 : {} : {}".format(len(sys.argv), sys.argv))
                log.info("[Check] 실행 방법 : {}".format('python3 DtaProcess.py argv1 argv2 ...'))
                sys.exit(0)

            log.info("[Check] sys.argv : {}".format(sys.argv))

            # breakpoint()
            # self.argv3, self.argv4 = map(lambda x: str(x), sys.argv[1:])

            for i, val in enumerate(sys.argv[1:]):
                key = 'argv{}'.format(i + 1)
                # globals()[key] = val
                setattr(self, key, val)

                log.info("[Check] {} : {}".format(key, val))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    def setEnv(self):
        log.info("[START] setEnv : {}".format("init"))

        try:
            global glVar

            # 작업환경 경로 설정
            contextPath = os.getcwd()

            # 전역 변수
            glVar = {
                "config": {
                    "imgContextPath": contextPath + '/../resources/image/'
                    , "csvConfigPath": contextPath + '/../resources/data/csv/'
                }
            }

            log.info("[Check] glVar : {}".format(glVar))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise

        finally:
            log.info("[END] setEnv : {}".format("init"))

    # ================================================================================================
    # 초기 변수 설정
    # ================================================================================================
    def setData(self):
        log.info("[START] setData : {}".format("init"))

        try:
            self.var = self.argv1
            self.var2 = self.argv2

            log.info("[Check] self.var : {}".format(self.var))
            log.info("[Check] self.var2  : {}".format(self.var2))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise

        finally:
            log.info("[END] setData : {}".format("init"))

    # ================================================================================================
    # 초기 함수 정의
    # ================================================================================================
    def setFun(self):
        log.info("[START] setFun : {}".format("init"))

        try:
            result = np.sum([int(self.var), int(self.var2)])

            return result
        except Exception as e:
            log.error("Exception : {}".format(e))
            raise

        finally:
            log.info("[END] setFun : {}".format("init"))

    # ================================================================================================
    # 초기 변수 및 함수를 통해 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):
        log.info("[START] exec : {}".format("init"))

        try:
            result = self.setFun()

            log.info("[Check] result : {}".format(result))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise

        finally:
            log.info("[END] exec : {}".format("init"))


if __name__ == '__main__':

    try:
        callDtaProcess = DtaProcess()

        # 초기 환경변수 설정
        callDtaProcess.setEnv()

        # 초기 변수 설정
        callDtaProcess.setData()

        # 초기 변수 및 함수를 통해 비즈니스 로직 수행
        callDtaProcess.exec()

    except Exception as e:
        log.error("Exception : {}".format(e))
        sys.exit(1)