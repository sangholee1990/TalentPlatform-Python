#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# bash /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-LSH0613.sh

#========================================
# Init Config
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
#CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/proj/indisystem/2024/SHELL
#CTX_PATH=/home/guest_user1/SYSTEMS/KIER
CTX_PATH=/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell

# 실행 파일 경로
PY36_PATH=/HDD/SYSTEMS/LIB/anaconda3/envs/py36
PY36_BIN=${PY36_PATH}/bin/python3

# 프로세스 종류
MULTI_PROC=10

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] TMP_PATH : $TMP_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] UPD_PATH : $UPD_PATH"

#========================================
# Argument Check
#========================================
#if [ "$#" -ne 2 ]; then
#
#   echo
#   echo "$# is Illegal Number of Arguments"
#
##   echo 'Example) bash '$0' "2023-08-28" "2023-08-30"'
##   echo 'Example) bash '$0' "2024-01-01 00:00" "2024-01-01 00:00"'
#   echo 'Example) bash '$0' "2024-01-01 00:00" "2024-01-02 00:00"'
#   echo 'Example) bash '$0' "2023-11-01 00:00" "2024-01-01 00:00"'
#   echo
#
#   exit
#fi
#
#srtDate="$1"
#endDate="$2"
#
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] srtDate : $srtDate"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] endDate : $endDate"

#========================================
# Run Shell
#========================================
cnt=0
fileList=$(find "/DATA/OUTPUT/LSH0613/전처리" -name "건축인허가-아파트전월세_*_*.csv" -type f 2>/dev/null | sort -u)
for fileInfo in $fileList; do
  sgg=$(echo "$fileInfo" | awk -F '[_.]' '{print $2, $3}')
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] cnt : $cnt / sgg : $sgg"

  ${PY36_BIN} /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-Model.py --searchSggList "$sgg"
  sleep 1s
  let cnt++

  if [ $cnt -ge ${MULTI_PROC} ]; then
    wait
    pkill -f h2o.jar
    cnt=0
  fi
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0