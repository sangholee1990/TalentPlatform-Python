# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
import warnings
import yaml
from sympy import exp

import common.initiator as common
from application import Application
import re
import os


def main():
    try:
        # option = get_option()
        # inFile = option.inFile
        # modelName = option.modelName.upper()
        inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc'
        modelName = 'KIER-LDAPS'

        ctxPath = os.getcwd()
        logInfo = f'{ctxPath}/log/daemon-kierDB-{modelName}.log'
        cfgInfo = f'{ctxPath}/config/config.yml'

        os.makedirs(os.path.dirname(logInfo), exist_ok=True)
        os.makedirs(os.path.dirname(cfgInfo), exist_ok=True)

        # common.init_logger("./logdir/gribDB.log")
        common.init_logger(logInfo)
        config = parse_config(cfgInfo)

        if re.search('wrfout', inFile, re.IGNORECASE):
            modelName1 = modelName + "_ALL"
        elif re.search('unis', inFile, re.IGNORECASE):
            modelName1 = modelName + "_UNIS"
        elif re.search('pres', inFile, re.IGNORECASE):
            modelName1 = modelName + "_PRES"
        else:
            modelName1 = modelName

        if not os.path.isfile(inFile):
            common.logger.error("[ERROR] File is not Exists")
            exit(1)

        common.logger.info(f'[CHECK] modelName : {modelName}')
        apps = Application(inFile, modelName, modelName1, config)
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
        parser.add_argument('modelName', type=str, help='Model name is None please check the argments')
        return parser.parse_args()
    except KeyError as e:
        common.logger.error("check the input argments (Input File, Model Name)")


if __name__ == '__main__':
    main()