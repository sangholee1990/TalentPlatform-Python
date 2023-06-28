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
    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'   # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0183'

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

            fileInfo1 = glob.glob('{}/{}'.format(globalVar['inpPath'], '/LSH0183/result/reply.csv'))
            if (len(fileInfo1) < 1): raise Exception("[ERROR] fileInfo1 : {}".format("자료를 확인해주세요."))

            replyData = (
                (
                        pd.read_csv(fileInfo1[0]) >>
                        dfply.mutate(
                            title = ''
                            , view = ''
                            , content=dfply.X.reply
                            , flag = 'reply'
                        )  >>
                        dfply.select(
                            dfply.X.idx_no
                            , dfply.X.title
                            , dfply.X.content
                            , dfply.X.nick
                            , dfply.X.date
                            , dfply.X.view
                            , dfply.X.flag
                            , dfply.X.thread
                        )
                )
            )

            contentInfo = glob.glob('{}/{}'.format(globalVar['inpPath'], '/LSH0183/INPUT/CONTENT_RESULT.xlsx'))
            if (len(contentInfo) < 1): raise Exception("[ERROR] contentInfo : {}".format("자료를 확인해주세요."))

            sheetList = ['황반변성', '비오뷰', '루센티스', '아일리아', '아바스틴']

            # breakpoint()

            # sheetInfo = sheetList[0]
            for sheetInfo in sheetList:
                log.info('[CHECK] sheetInfo : {}'.format(sheetInfo))

                keyData = (
                    (
                            pd.read_excel(contentInfo[0], sheet_name=sheetInfo) >>
                            dfply.filter_by(
                                dfply.X.flag == 'content'
                            ) >>
                            dfply.mutate(
                                thread = ''
                            )
                    )
                )

                data = pd.DataFrame()
                for i in range(len(keyData)):
                    keyDataL1 = (
                        (
                                keyData >>
                                dfply.filter_by(
                                    dfply.X.idx_no == keyData['idx_no'][i]
                                    , dfply.X.view != None
                                ) >>
                                dfply.mutate(
                                    url=(
                                        "https://cafe.naver.com/maculardegeneration?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D21788988%2526page%3D1%2526boardtype%3DL%2526articleid%3D{}%2526referrerAllArticles%3Dtrue").format(
                                        keyData['idx_no'][i])
                                )
                        )
                    )

                    replyDataL1 = (
                        (
                                replyData >>
                                dfply.filter_by(
                                    dfply.X.idx_no == keyData['idx_no'][i]
                                    , dfply.X.thread != ''
                                ) >>
                                dfply.mutate(
                                    url = ''
                                )
                        )
                    )

                    # 행 단위로 추가
                    data = pd.concat([data, keyDataL1, replyDataL1], axis = 0)

                saveFile = '{}/{}_키워드_{}.xlsx'.format(globalVar['outPath'], serviceName, sheetInfo)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

                data.to_excel(saveFile, index=False)

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