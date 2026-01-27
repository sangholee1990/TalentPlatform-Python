#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo


#========================================
# DOC
#========================================
# * * * * * bash /SYSTEMS/PROG/SHELL/RunShell-ProcAgentCheck.sh
# tail -f /SYSTEMS/PROG/SHELL/LOG/*.log

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
#CTX_PATH=/SYSTEMS/PROG/SHELL/PROC
CTX_PATH=/SYSTEMS/PROG/SHELL

declare -A metaData
#metaData["TalentPlatform-QUBE2025-repro"]="/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-repro.py"
metaData["TalentPlatform-QUBE2025-repro2"]="/SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-repro2.py"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] LOG_PATH : $LOG_PATH"
echo

#========================================
# Run Shell
#========================================
for key in "${!metaData[@]}"; do
  val="${metaData[$key]}"
  #echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] key : $key / val : $val"

  procId=$(pgrep -f "$key")

  LOG_PATH=${CTX_PATH}/LOG/${key}
  LOG_NAME=$(basename "$0" .sh)_${key}_$(date +"%Y%m%d").log
  mkdir -p ${LOG_PATH}

  if [ -z "${procId}" ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running : $key"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running : $key" >> ${LOG_PATH}/${LOG_NAME}
      nohup bash -c "$val" >> ${LOG_PATH}/${LOG_NAME} 2>&1 &
  else
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: $key : ${procId}"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: $key : ${procId}" >> ${LOG_PATH}/${LOG_NAME}
  fi
done

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0

