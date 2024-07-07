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

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell
# nohup bash /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-LSH0454.sh &
# tail -f nohup.out

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
CTX_PATH=/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell
RUN_PATH=/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
#CTX_PATH=/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell
#RUN_PATH=/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys

COL_NAME=TalentPlatform-LSH0454-Active-OpenAPI-DataCollectToPrep.py
MOD_NAME=TalentPlatform-LSH0454-Active-OpenAPI-Model.py

PY36_BIN=/SYSTEMS/LIB/anaconda3/envs/py36/bin/python
PY38_BIN=/SYSTEMS/LIB/anaconda3/envs/py36/bin/python
#PY36_BIN=/HDD/SYSTEMS/LIB/anaconda3/envs/py36/bin/python
#PY38_BIN=/HDD/SYSTEMS/LIB/anaconda3/envs/py36/bin/python

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
# 2024.04.01
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
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 중랑구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 노원구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 도봉구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 종로구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 관악구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 중구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 동대문구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 서대문구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 마포구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 금천구"
#${PY_PATH} ${RUN_PATH}/${RUN_NAME} --addrList "서울특별시 은평구"

addrList=(
#    "경기도 과천시"
#    "서울특별시 용산구"
#    "서울특별시 송파구"
#    "서울특별시 서초구"
#    "서울특별시 강남구"
#    "서울특별시 강북구"
    "서울특별시 양천구"
#    "경기도 의정부시"
#    "경기도 하남시"
#    "서울특별시 성동구"
#    "서울특별시 광진구"
#    "서울특별시 중랑구"
#    "서울특별시 노원구"
#    "서울특별시 도봉구"
#    "서울특별시 종로구"
#    "서울특별시 관악구"
#    "서울특별시 중구"
#    "서울특별시 동대문구"
#    "서울특별시 서대문구"
#    "서울특별시 마포구"
#    "서울특별시 금천구"
#    "서울특별시 은평구"
)

for addr in "${addrList[@]}"; do
	  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] addr : $addr"

#    fileListCnt=$(find -L ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u | wc -l)
#    if [ ${fileListCnt} -lt 1 ]; then continue; fi

    # 수집/가공
#    ${PY38_BIN} ${RUN_PATH}/${COL_NAME} --addrList "$addr"

    # 예측
    ${PY36_BIN} ${RUN_PATH}/${MOD_NAME} --addrList "$addr"

done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0