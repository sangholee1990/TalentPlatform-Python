#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
PY38_PATH=/usr/local/anaconda3/envs/py38/bin/python3.8
CTX_PATH=$(pwd)
RUN_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "2022-07-01" "2022-12-31"'
   echo

   exit
fi

srtDate="$1"
endDate="$2"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] srtDate : $srtDate"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] endDate : $endDate"

#========================================
# Run Shell
#========================================
${PY38_PATH} ${RUN_PATH}/TalentPlatform-LSH0455.py --srtDate "$srtDate" --endDate "$endDate" --satList "OCO2-CO2" --shpInfo "gadm36_KOR_2" &
sleep 1s

${PY38_PATH} ${RUN_PATH}/TalentPlatform-LSH0455.py --srtDate "$srtDate" --endDate "$endDate" --satList "OCO3-CO2" --shpInfo "gadm36_KOR_2" &
sleep 1s

${PY38_PATH} ${RUN_PATH}/TalentPlatform-LSH0455.py --srtDate "$srtDate" --endDate "$endDate" --satList "OCO2-SIF" --shpInfo "gadm36_KOR_2" &
sleep 1s

${PY38_PATH} ${RUN_PATH}/TalentPlatform-LSH0455.py --srtDate "$srtDate" --endDate "$endDate" --satList "OCO3-SIF" --shpInfo "gadm36_KOR_2" &
sleep 1s

${PY38_PATH} ${RUN_PATH}/TalentPlatform-LSH0455.py --srtDate "$srtDate" --endDate "$endDate" --satList "TROPOMI" --shpInfo "gadm36_KOR_2" &
sleep 1s

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0