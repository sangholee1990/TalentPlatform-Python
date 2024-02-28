# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
import warnings
import yaml
import common.initiator as common
from application import Application
import re
import os

def main():
    try:
        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc'
        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc'
        # modelName = 'KIER-LDAPS-2K'
        # modelName = 'KIER-LDAPS-2K-ORG'
        # modelName = 'KIER-LDAPS-2K-30M'
        # modelName = 'KIER-LDAPS-2K-60M'

        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc'
        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc'
        # modelName = 'KIER-RDAPS-3K'
        # modelName = 'KIER-RDAPS-3K-ORG'
        # modelName = 'KIER-RDAPS-3K-30M'
        # modelName = 'KIER-RDAPS-3K-60M'

        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIM/r030_v040_ne36_pres_h006.2023063000.gb2'
        # inFile = '/DATA/INPUT/INDI2023/MODEL/KIM/r030_v040_ne36_unis_h001.2023063000.gb2'
        # modelName = 'KIM-3K'

        # inFile = '/DATA/INPUT/INDI2023/MODEL/LDAPS/l015_v070_erlo_pres_h024.2023062918.gb2'
        # inFile = '/DATA/INPUT/INDI2023/MODEL/LDAPS/l015_v070_erlo_unis_h024.2023062918.gb2'
        # modelName = 'LDAPS-1.5K'

        # inFile = '/DATA/INPUT/INDI2023/MODEL/RDAPS/g120_v070_erea_pres_h024.2023062918.gb2'
        # inFile = '/DATA/INPUT/INDI2023/MODEL/RDAPS/g120_v070_erea_unis_h024.2023062918.gb2'
        # modelName = 'RDAPS-12K'

        # inFile = '/DATA/INPUT/INDI2023/DATA/GFS/2023/08/28/00/gfs.t00z.pgrb2.0p25.f003.gb2'
        # modelName = 'GFS-25K'

        # inFile = '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/2024/01/01/reanaly-era5-pres_202401010600.nc'
        # inFile = '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/2024/01/01/reanaly-era5-unis_202401010600.nc'
        # modelName = 'REANALY-ERA5-25K'

        # inFile = '/DATA/INPUT/INDI2024/DATA/SAT-SENT1/2024/01/19/S1A_ESA_2024_01_19_09_32_41_0758971961_126.46E_37.99N_VV_C11_GFS025CDF_wind.png'
        # modelName = 'SAT-SENT1'

        inFile = '/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00'
        modelName = 'KIER-LDAPS-0.6K-ORG'
        # modelName = 'KIER-LDAPS-0.6K-10M'
        # modelName = 'KIER-LDAPS-0.6K-30M'
        # modelName = 'KIER-LDAPS-0.6K-60M'

        # option = get_option()
        # inFile = option.inFile
        # modelName = option.modelName

        ctxPath = os.getcwd()
        logInfo = f'{ctxPath}/log/daemon-kierDB-{modelName}.log'
        cfgInfo = f'{ctxPath}/config/config.yml'

        os.makedirs(os.path.dirname(logInfo), exist_ok=True)
        os.makedirs(os.path.dirname(cfgInfo), exist_ok=True)

        # common.init_logger("./logdir/gribDB.log")
        common.init_logger(logInfo)
        config = parse_config(cfgInfo)

        common.logger.info(f'[CHECK] ctxPath : {ctxPath}')
        common.logger.info(f'[CHECK] logInfo : {logInfo}')
        common.logger.info(f'[CHECK] cfgInfo : {cfgInfo}')

        modelKey = ""
        if re.search('wrfout_d01|wrfout_d04|gfs|S1A_ESA', inFile, re.IGNORECASE):
            modelKey = "ALL"
        elif re.search('unis|wrfsolar_d02', inFile, re.IGNORECASE):
            modelKey = "UNIS"
        elif re.search('pres|wrfout_d02', inFile, re.IGNORECASE):
            modelKey = "PRES"

        if not os.path.isfile(inFile):
            common.logger.error(f"입력 파일 ({inFile})을 확인해주세요.")
            exit(1)

        if not (
                (re.search('wrfout_d01|wrfout_d04|gfs', inFile, re.IGNORECASE) and re.search('ALL', modelKey, re.IGNORECASE))
                or (re.search('unis|wrfsolar_d02', inFile, re.IGNORECASE) and re.search('UNIS', modelKey,re.IGNORECASE))
                or (re.search('pres|wrfout_d02', inFile, re.IGNORECASE) and re.search('PRES', modelKey, re.IGNORECASE))
        ):
            common.logger.error(f'입력 파일 ({inFile}) 및 모델 키({modelKey})를 확인해주세요.')
            exit(1)

        common.logger.info(f'[CHECK] modelName : {modelName}')
        common.logger.info(f'[CHECK] modelKey : {modelKey}')

        apps = Application(inFile, modelName, modelKey, config)
        apps.run()

    except Exception as e:
        common.logger.error(f'[ERROR] check the argments or data type or path : {e}')


def parse_config(configPath):
    """
    설정파일 파싱
    :param config_path: 설정파일 경로
    :return: 설정파일 파싱 결과
    """
    with open(configPath, "rt", encoding="utf-8") as stream:
        parseConfigResult = yaml.safe_load(stream)
        return parseConfigResult


def get_option():
    """
    모듈 실행에 필요한 파라미터 파싱 함수
    """
    try:
        parser = argparse.ArgumentParser(description='generate data')
        parser.add_argument('inFile', type=str, help='input File is None please check the argments')
        parser.add_argument('modelName', type=str, help='Model Name is None please check the argments')
        return parser.parse_args()
    except KeyError as e:
        common.logger.error("check the input argments (Input File, Model Name)")

if __name__ == '__main__':
    main()