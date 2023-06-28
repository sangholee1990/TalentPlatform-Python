# -*- coding: utf-8 -*-

# from plotnine import *
# from plotnine.data import *
# from dfply import *
# import hydroeval
import dfply
# from keras.layers import LSTM
# from keras.models import Sequential
#
# from keras.layers import Dense
# import keras.backend as K
# from keras.callbacks import EarlyStopping
import traceback

# 초기 환경변수 정의
from src.talentPlatform.unitSys.helper.InitConfig import *

class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 데이터 전처리

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'   # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0184'

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

            fileInfoPattrn = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '1.csv')
            fileInfo = glob.glob(fileInfoPattrn)
            if (len(fileInfo) < 1): raise Exception("[ERROR] fileInfo : {} : {}".format("자료를 확인해주세요.", fileInfoPattrn))

            data = pd.read_csv(fileInfo[0], skiprows = 16)

            # (Pdb) >? dataL1.columns
            # Index(['DateTime', 'Latitude', 'L Sensing Latitude', 'R Sensing Latitude',
            #        'Longitude', 'L Sensing Longitude', 'R Sensing Longitude',
            #        'Sensor R S1', 'Sensor L S1', 'Cropspec Root S1'],
            #       dtype='object')

            dataL1 = (
                    (
                        data >>
                        dfply.select(
                            dfply.X['DateTime']
                            , dfply.X['Latitude']
                            , dfply.X['Longitude']
                            , dfply.X['Cropspec Root S1']

                            , dfply.X['L Sensing Latitude']
                            , dfply.X['L Sensing Longitude']
                            , dfply.X['Sensor L S1']

                            , dfply.X['R Sensing Latitude']
                            , dfply.X['R Sensing Longitude']
                            , dfply.X['Sensor R S1']
                        )
                    )
            )

            dataL2 = dataL1.replace(0, np.nan)\
                .dropna(axis = 0)

            dataL3 = pd.concat([
                pd.DataFrame(dataL2[['DateTime', 'Latitude', 'Longitude', 'Cropspec Root S1']]).set_axis(['DateTime', 'y', 'x', 'S1'], axis = 1, inplace=False)
                , pd.DataFrame(dataL2[['DateTime', 'L Sensing Latitude', 'L Sensing Longitude', 'Sensor L S1']]).set_axis(['DateTime', 'y', 'x', 'S1'], axis = 1, inplace=False)
                , pd.DataFrame(dataL2[['DateTime', 'R Sensing Latitude', 'R Sensing Longitude', 'Sensor R S1']]).set_axis(['DateTime', 'y', 'x', 'S1'], axis = 1, inplace=False)
            ]
                ,  axis = 0
            )

            dataL4 = dataL3.sort_values(by=['DateTime'], axis=0)

            saveFile = '{}/{}_{}'.format(globalVar['outPath'], serviceName, '2021_nagano_S1_01_raw.csv')
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            dataL4.to_csv(saveFile, index=False)

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