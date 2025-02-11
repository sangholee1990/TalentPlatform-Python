#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo


#========================================
# DOC
#========================================
#cp -f /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-ProcAgentCheck.sh /SYSTEMS/PROG/SHELL/PROC/.
#bash /SYSTEMS/PROG/SHELL/PROC/RunShell-ProcAgentCheck.sh

#ps -ef | grep "TalentPlatform-LSH0605-DaemonFramework" | awk '{print $2}' | xargs kill -9
#ps -ef | grep "chrome" | awk '{print $2}' | xargs kill -9

# * * * * * bash /SYSTEMS/PROG/SHELL/PROC/RunShell-ProcAgentCheck.sh

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
CTX_PATH=/SYSTEMS/PROG/SHELL/PROC

declare -A metaData
#metaData["SOLARMY_APP"]="/SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB/build/SOLARMY_APP"
metaData["TalentPlatform-LSH0605-DaemonFramework"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py"

LOG_PATH=${CTX_PATH}/LOG
LOG_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d").log
mkdir -p ${LOG_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] LOG_PATH : $LOG_PATH"
echo

#========================================
# Run Shell
#========================================
for key in "${!metaData[@]}"; do
  val="${metaData[$key]}"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] key : $key / val : $val"

  procId=$(pgrep -f "$key")

  if [ -z "${procId}" ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running" >> ${LOG_PATH}/${LOG_NAME}
      nohup $val >> ${LOG_PATH}/${LOG_NAME} 2>&1 &
  else
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: ${procId}"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: ${procId}" >> ${LOG_PATH}/${LOG_NAME}
  fi
done

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0

