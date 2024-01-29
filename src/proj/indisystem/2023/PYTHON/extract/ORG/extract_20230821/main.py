# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime
import warnings
import yaml
import common.initiator as common
from application import Application
import re

# python main.py /vol01/DATA/MODEL/KIM/r030_v040_ne36_pres_h006.2023063000.gb2 KIM
# python main.py /vol01/DATA/MODEL/KIM/r030_v040_ne36_unis_h001.2023063000.gb2 KIM
# #python main.py /vol01/DATA/MODEL/LDAPS/l015_v070_erlo_pres_h024.2023062918.gb2 LDAPS
# #python main.py /vol01/DATA/MODEL/LDAPS/l015_v070_erlo_unis_h024.2023062918.gb2 LDAPS
# python main.py /vol01/DATA/MODEL/RDAPS/g120_v070_erea_pres_h024.2023062918.gb2 RDAPS
# python main.py /vol01/DATA/MODEL/RDAPS/g120_v070_erea_unis_h024.2023062918.gb2 RDAPS

# python3 main.py /vol01/DATA/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc KIER-RDAPS
# python3 main.py /vol01/DATA/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc KIER-RDAPS
# python3 main.py /vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc KIER-LDAPS
# python3 main.py /vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc KIER-LDAPS

def main () :
	try:
		# option = get_option()
		# inFile = option.inFile
		# modelName = option.modelName.upper()
		# inFile = '/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc'
		inFile = '/vol01/DATA/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc'
		modelName = 'KIER-RDAPS'

		# common.init_logger(./logdir/gribDB.log")
		# configPath="./config/config.yml"
		ctxPath = '/vol01/SYSTEMS/KIER/PROG/PYTHON/extract_20230821'

		common.init_logger(f'{ctxPath}/logdir/gribDB.log')
		configPath=f'{ctxPath}/config/config.yml'

		config = parse_config(configPath)
		if re.search('unis|wrfsolar', inFile, re.IGNORECASE):
			modelName1 = modelName + "_UNIS"
		elif re.search('pres|wrfout', inFile, re.IGNORECASE):
			modelName1 = modelName + "_PRES"
		else:
			modelName1 = modelName

		if os.path.isfile(inFile) :
#			varNameLists= config['modelName'][modelName1]['varName']
#			locNameLists= config['modelName'][modelName1]['locName']
#			levelLists= config['modelName'][modelName1]['hPaLevel']
#			apps = Application(inFile,modelName,config,varNameLists,locNameLists,levelLists)    
			apps = Application(inFile, modelName, modelName1, config)
			apps.run()
		else:
			common.logger.error("[ERROR]File is not Exists")
			exit()

	except KeyError as e:
		common.logger.error("check the argments or data type or path" + e)      

def parse_config(configPath):
    """
    설정파일 파싱
    :param config_path: 설정파일 경로
    :return: 설정파일 파싱 결과
    """
    with open(configPath, "rt", encoding="utf-8") as stream:
        parseConfigResult = yaml.safe_load(stream)
        return parseConfigResult
    
def get_option() :
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

if __name__ =='__main__':
    main()
