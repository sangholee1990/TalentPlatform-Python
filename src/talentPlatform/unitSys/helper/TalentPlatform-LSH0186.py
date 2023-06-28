# -*- coding: utf-8 -*-

# from plotnine import *
# from plotnine.data import *
# from dfply import *
# import hydroeval
# from keras.layers import LSTM
# from keras.models import Sequential
#
# from keras.layers import Dense
# import keras.backend as K
# from keras.callbacks import EarlyStopping
import traceback

# 초기 환경변수 정의
from src.talentPlatform.unitSys.helper.InitConfig import *

from tensorflow.keras.layers import Conv2D, add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras import Input
# import cv2
import numpy as np


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 천리안위성 해수면온도 CNN-PCA와 LSTM 예측

    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0186'

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

            # fileInfoPattrn = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '1.csv')
            # fileInfo = glob.glob(fileInfoPattrn)
            # if (len(fileInfo) < 1): raise Exception("[ERROR] fileInfo : {} : {}".format("자료를 확인해주세요.", fileInfoPattrn))
            # saveFile = '{}/{}_{}'.format(globalVar['figPath'], serviceName, '2021_nagano_S1_01_raw.png')
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # breakpoint()

            fileInfoPattrn = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'coms_data/coms_mi_le2_sst_data1.txt')
            fileInfo = glob.glob(fileInfoPattrn)
            if (len(fileInfo) < 1): raise Exception("[ERROR] fileInfo : {} : {}".format("자료를 확인해주세요.", fileInfoPattrn))

            data = pd.read_csv(fileInfo[0], header=None)

            # 이미지 그리기
            fileName = os.path.basename(fileInfo[0])
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileName)

            plt.pcolormesh(data)
            plt.clim()
            plt.colorbar()
            plt.savefig(saveImg, dpi=600, bbox_inches='tight')
            plt.show()

            # patch_size = (33, 33)
            patch_size = data.shape

            input_shape = (patch_size[0], patch_size[1], 1)
            batch_size = 64

            input_img = Input(shape=input_shape)

            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)

            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)

            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)

            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            res_img = model

            output_img = add([res_img, input_img])

            model = Model(input_img, output_img)

            model.compile(loss=MeanSquaredError(),
                          optimizer=Adam(),
                          metrics=['accuracy'])
                          # metrics=[PSNRLoss])

            # breakpoint()


            # img = cv2.imread('test.jpg')
            # img = cv2.resize(img, (320, 240))
            # img = np.reshape(data, [patch_size[0], patch_size[1]])

            img = np.expand_dims(data, axis=0)
            predict = model.predict(img)

            # # 이미지 그리기
            # plt.pcolormesh(predict)
            # plt.clim()
            # plt.colorbar()
            # plt.show()
            #
            # breakpoint()

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