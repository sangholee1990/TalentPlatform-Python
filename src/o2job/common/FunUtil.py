
# 라이브러리 읽기
import json
import json
import logging as log
import math
import os
import os
import os
import shutil
# import salt.client
# import salt.runner
import subprocess
import sys
import sys
import traceback
import traceback
import warnings
import ConfigParser
import numpy as np
import pysftp
# import StringIO
import requests

# 로그 설정
log.basicConfig(stream=sys.stdout, level=log.INFO,
                format="%(asctime)s [%(filename)s > %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
warnings.filterwarnings("ignore")

# ================================================================================================
# 초기 변수 정의
# ================================================================================================
restUrl = "http://das-sv-gis-2:9101/geoserver"
auth = ('admin', 'geoserver')
headers = {'Content-Type': 'application/json'}

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


"""
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        디렉토리 경로 이동 처리. 대상 경로가 있을 경우 삭제한 다음 이동 처리 수행
"""
def relocPath(src, dst):
    log.info("[START] relocPath : {}".format("init"))

    try:
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    except Exception as e:
        log.error("Exception : {}".format(e))
        raise

    finally:
        log.info("[END] relocPath : {}".format("init"))

"""
    parameter:
	dst - 대상 경로
	uid - OS 유저 ID
	gid - OS 그룹 ID
    return:         
        none
    description: 
        대상 경로 하위의 모든 파일에 대하여 일괄 소유자 변경 처리 수행
"""
def changeOwner(dst, uid, gid):
    log.info("[START] changeOwner : {}".format("init"))

    try:
        os.chown(dst, uid, gid)

        for fn in os.listdir(dst):
            path = dst + "/" + fn
            os.chown(path, uid, gid)
    except Exception as e:
        log.error("Exception : {}".format(e))
        raise

    finally:
        log.info("[END] changeOwner : {}".format("init"))

"""
    parameter:
	path - 생성 경로
	isdir - 디렉토리 여부
	mod - 권한 정보
    return:         
        none
    description: 
        대상 경로 생성 처리 수행. 대상 경로가 존재할 경우 삭제 후 생성.
"""
def makeDirs(path, isdir=True, mod=None):
    log.info("[START] copyDir : {}".format("init"))

    try:
        dir = path
        if not isdir:
            dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
            if mod:
                os.chmod(dir, mod)
    except Exception as e:
        log.error("Exception : {}".format(e))
        raise

    finally:
        log.info("[END] makeDirs : {}".format("init"))

"""
    parameter:
	strs - 경로 문자 배열
    return:         
        string - 전체 경로 문자열
    description: 
        해당 문자열에 해당하는 디렉토리 경로를 문자열을 생성
"""
def concatPath(strs):
    log.info("[START] concatPath : {}".format("init"))

    try:
        fullPath = ""

        for str in strs:
            if not str: continue
            if str == '': continue
            fullPath = fullPath + "/" + str
        return fullPath

    except Exception as e:
        log.error("Exception : {}".format(e))
        raise

    finally:
        log.info("[END] concatPath : {}".format("init"))

"""
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        원본 경로의 디렉토리를 대상 경로로 복사. 이미 대상 경로가 있을 경우 삭제 후 처리
"""
def copyDir(src, dst):
    log.info("[START] copyDir : {}".format("init"))

    try:

        makeDirs(dst)

        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    except Exception as e:
        log.error("Exception : {}".format(e))
        raise

    finally:
        log.info("[END] copyDir : {}".format("init"))

"""
    module:
        commonUtil
    function:
     	copyFile
    parameter:
	src - 원본 경로
	dst - 대상 경로
    return:         
        none
    description: 
        원본 경로의 파일을 대상 경로에 복사. 이미 대상 파일이 있을 경우 삭제 후 처리
"""
def copyFile(src, dst):
    dstDir = os.path.dirname(dst)
    makeDirs(dstDir)

    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)

"""
    module:
        commonUtil
    function:
     	loadModule
    parameter:
	modname - 모듈명
	fromlist - 패키지명
    return:         
        none
    description: 
        python 모듈 동적 로딩 처리 수행
"""
def loadModule(modname,fromlist):
    mod = None
    try:
        mod = __import__(modname, fromlist=fromlist)
    except:
        pass
    return mod

"""
    module:
        commonUtil
    function:
     	ftpIsDirectory
    parameter:
	ftp - ftp 객체
	dir - ftp 대상 경로
    return:         
        bool - 디렉토리 여부 값
    description: 
        ftp 의 해당 경로가 존재하는지 여부 확인
"""
def ftpIsDirectory(ftp, dir):
    try:
        ftp.cwd(dir)
    except:
        return False
    return True

"""
    module:
        commonUtil
    function:
     	sftpIsDirectory
    parameter:
	sftp - sftp 객체
	dir - sftp 대상 경로
    return:         
        bool - 디렉토리 여부 값
    description: 
        sftp 의 해당 경로가 존재하는지 여부 확인
"""
def sftpIsDirectory(sftp, dir):
    try:
        sftp.stat(dir)
    except:
        return False
    return True

"""
    module:
        commonUtil
    function:
     	ftpMakeDirs
    parameter:
	ftp - ftp 객체
	dir - ftp 대상 경로
    return:         
        none
    description: 
        ftp 의 대상 경로를 생성
"""
def ftpMakeDirs(ftp, dir):
    dirs = dir.split("/")

    for i in range( 1, len(dirs) + 1 ):
        d = "/".join(dirs[0:i])
        if not ftpIsDirectory(ftp, d):
            ftp.mkd(d)

"""
    module:
        commonUtil
    function:
     	getDirFileSize
    parameter:
	rootDir - 대상 경로
    return:         
        int - 파일 사이즈 총합 (바이트 단위)
    description: 
        대상 경로 하위의 파일 사이즈 계산
"""
def getDirFileSize(rootDir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(rootDir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

"""
    module:
        commonUtil
    function:
     	getDirFileCount
    parameter:
	rootDir - 대상 경로
    return:         
        int - 파일 수량 총합
    description: 
        대상 경로 하위의 파일 수량 계산
"""
def getDirFileCount(rootDir):
    total_cnt = 0
    for dirpath, dirnames, filenames in os.walk(rootDir):
        for f in filenames:
            total_cnt += 1
    return total_cnt

"""
    module:
        commonUtil
    function:
     	getMatchFilename
    parameter:
	rootDir - 대상 경로
	exts - 확장자 배열
    return:         
        string - 파일명
    description: 
        대상 경로 하위에 해당하는 확장자가 매칭되는 파일 검색
"""
def getMatchFilename(rootDir, exts):
    for filenm in os.listdir(rootDir):
        fn, ext = os.path.splitext(filenm)
        if ext in exts:
            return filenm

"""
    module:
        commonUtil
    function:
     	isValidate
    parameter:
	dst - 대상 경로
    return:         
    	bool - 유효성 여부
    description: 
        대상 경로 하위의 파일 사이즈로부터 유효성 검사 수행
"""
def isValidate(dst):
    if os.path.isdir(dst):
        for fn in os.listdir(dst):
            path = dst + "/" + fn
            if os.stat(path).st_size <= 1:
                return False
    else:
        if os.stat(dst).st_size <= 1:
            return False

    return True

def loadConfig(self, confFile):
    try:
        if not os.path.exists(confFile):
            raise Exception("%s file not found..\n" %confFile)
        self.config = ConfigParser.ConfigParser()
        self.config.read(confFile)
    except(Exception, OSError):
        raise("load Configuration file error")

def getValue(self, section, key):
    value = self.config.get(section, key)
    return value

def getJson(self, section, key):
    value = self.config.get(section, key)
    j = json.loads(value)
    return j



"""
    module:
        reggeo
    function:
     	createLayer
    parameter:
	prodId - 산출 자료 ID
	tifPath - TIF 파일 경로
	defstyle - Geoserver 스타일명
    return:         
    	none
    description: 
        TIF 파일을 기준으로 Geoserver 에 레이어 등록 및 스타일링 처리 수행
"""
def createLayer(prodId, tifPath, defstyle='lut_omi'):
    # log = logUtil.initLog("backend")

    # delete coveragestore
    url = restUrl + "/rest/workspaces/img/coveragestores/"+prodId+"?recurse=true"
    r = requests.delete(url, headers=headers, auth=auth)
    log.info("delete coveragestore: " + str(r))

    # create coveragestore/store
    srcPath = ("file://" + tifPath)

    url = restUrl + "/rest/workspaces/img/coveragestores.json"
    data = {'coverageStore':{
            'name':prodId,'type':'GeoTIFF','enabled':'true',
            'workspace':'img','url':srcPath}}
    r = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
    log.info("create coveragestore: " + str(r))
    if not '[20' in str(r):
        raise Exception("CREATE COVERAGESTORE FAILED")

    # create coverage
    url = restUrl + "/rest/workspaces/img/coveragestores/"+prodId+"/coverages.json"
    data = {'coverage':{
            'name':prodId,'nativeCoverageName':prodId}}
    r = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
    log.info("create coverage: " + str(r))
    if not '[20' in str(r):
        raise Exception("CREATE COVERAGE FAILED")

    # apply layer style
    if defstyle:
        url = restUrl + "/rest/layers/img:"+prodId+".json"
        data = {'layer':{'defaultStyle':{'name':defstyle}}}
        r = requests.put(url, headers=headers, auth=auth, data=json.dumps(data))
        log.info("apply layer style: " + str(r))
        if not '[20' in str(r):
            raise Exception("APPLY LAYER STYLE FAILED")