#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
PY38_PATH=/SYSTEMS/anaconda3/envs/py36/bin/python3.6
#CTX_PATH=$(pwd)
CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell
RUN_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
RUN_NAME=TalentPlatform-LSH0454-Active-OpenAPI-Model.py

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"

#========================================
# Argument Check
#========================================
#if [ "$#" -ne 2 ]; then
#
#   echo
#   echo "$# is Illegal Number of Arguments"
#
#   echo 'Example) bash '$0' "2022-07-01" "2022-12-31"'
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
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 과천시"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 용산구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 송파구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 서초구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 강남구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 강북구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 양천구"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 의정부시"
${PY38_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 하남시"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0