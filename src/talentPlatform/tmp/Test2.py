
# if __name__ == '__main__':
# import InitConfig.globalVar as globalVar


# from src.talentPlatform.InitConfig import log, globalVar
from src.talentPlatform.unitSysHelper.InitConfig import *

# prjName = "asdasdasdasdasdasd"


# from src.talentPlatform.InitConfig import *
# from src.talentPlatform.InitConfig import globalVar
# from src.talentPlatform.InitCConfig import log
#

contextPath = 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
prjName = '99999999'
log = initLog(prjName)
globalVar = initGlobalVar(contextPath, prjName)


log.info("[ERROR]")
log.info(globalVar)