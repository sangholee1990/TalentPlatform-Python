#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# ps -ef | grep python3 | grep LSH0465 | awk '{print $2}' | xargs kill -9
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell
# nohup bash RunShell-LSH0465-PROC.sh &

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
CTX_PATH=$(pwd)
RUN_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
PY38_BIN=/SYSTEMS/anaconda3/envs/py38/bin/python3

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"

#========================================
# Argument Check
#========================================
#if [ "$#" -ne 2 ]; then
#
#   echo
#   echo "$# is Illegal Number of Arguments"
#
#   echo 'Example) bash '$0' "1979-01-01" "2020-01-01"'
#   echo
#
#   exit
#fi
#
#
#srtDate="$1"
#endDate="$2"
#
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] srtDate : $srtDate"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] endDate : $endDate"

#========================================
# Run Shell
#========================================
addList=("대전광역시" "울산광역시" "세종특별자치시" "경기도" "강원특별자치도" "충청북도" "충청남도" "전라북도" "전라남도" "경상북도" "경상남도" "제주특별자치도" "서울특별시" "부산광역시" "대구광역시" "인천광역시" "광주광역시")

for addrInfo in "${addList[@]}"
do
    ${PY38_BIN} ${RUN_PATH}/TalentPlatform-LSH0465.py --addrList "$addrInfo" &
    sleep 1s
done

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0