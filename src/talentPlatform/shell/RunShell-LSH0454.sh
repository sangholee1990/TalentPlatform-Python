#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# 전처리 파일 업로드
# /DATA/OUTPUT/LSH0454/전처리

# 예측 파일 다운로드
# /DATA/OUTPUT/LSH0454/예측

# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell
# nohup bash /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell/RunShell-LSH0454.sh &

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell
RUN_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
RUN_NAME=TalentPlatform-LSH0454-Active-OpenAPI-Model.py
PY_PATH=/SYSTEMS/anaconda3/envs/py36/bin/python3.6

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
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 과천시"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 용산구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 송파구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 서초구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 강남구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 강북구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 양천구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 의정부시"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "경기도 하남시"

# 2024.04.04
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 성동구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 광진구"

# 2024.04.07
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 중랑구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 노원구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 도봉구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 종로구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 관악구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 중구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 동대문구"
${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 서대문구"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0