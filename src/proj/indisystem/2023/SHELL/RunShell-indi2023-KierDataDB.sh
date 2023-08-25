#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
CTX_PATH=$(pwd)

# 아나콘다 가상환경 활성화
source /home/indisystem/.bashrc
conda activate py38

#PY38_BIN=/vol01/SYSTEMS/KIER/LIB/anaconda3/envs/py38/bin/python3.8
PY38_BIN=/home/indisystem/.conda/envs/py38/bin/python3
RUN_PATH=/vol01/SYSTEMS/KIER/PROG/PYTHON/extract
CFG_INFO=${RUN_PATH}/config/config.yml

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CFG_INFO : $CFG_INFO"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "2023-06-29" "2023-07-01"'
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
${PY38_BIN} ${RUN_PATH}/TalentPlatform-indi2023-KierDataDB.py --srtDate "$srtDate" --endDate "$endDate" --cfgInfo "$CFG_INFO"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0