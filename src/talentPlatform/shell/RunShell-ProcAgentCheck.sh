#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo


#========================================
# DOC
#========================================
# * * * * * bash /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-ProcAgentCheck.sh
# pkill -f SOLARMY_APP && cd /SYSTEMS/IOT/Roverdyn/daemon-iot-tcpipDb/build && cmake .. && make && bash /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-ProcAgentCheck.sh
# tail -f /SYSTEMS/PROG/SHELL/PROC/LOG/*.log

# pkill -f SOLARMY_APP
# pkill -f TCP-Server-Bees
# pkill -f TalentPlatform-bdwide-FileWatch

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
#CTX_PATH=/SYSTEMS/PROG/SHELL/PROC
CTX_PATH=/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell

declare -A metaData
#metaData["SOLARMY_APP"]="/SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB/build/SOLARMY_APP"
#metaData["TalentPlatform-LSH0605-DaemonFramework.py"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py"
#metaData["TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-DataCollectToPrep.py"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-DataCollectToPrep.py"
#metaData["TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-Model.py"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py36/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0613-DaemonFramework-Active-OpenAPI-Model.py"
#metaData["open-webui"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/open-webui serve --host 0.0.0.0 --port 9310"
#metaData["open-webui"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/open-webui serve --host 0.0.0.0 --port 9210"
#metaData["port 9099"]="bash /DATA/TMP/pipelines/start.sh"
metaData["nginx"]="systemctl restart nginx"
metaData["SOLARMY_APP"]="/SYSTEMS/IOT/Roverdyn/daemon-iot-tcpipDb/build/SOLARMY_APP"
metaData["TCP-Server-Bees"]="/SYSTEMS/IOT/Roverdyn/bee-iot/TCP-Server-Bees"

metaData["open-webui"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/open-webui serve --host 0.0.0.0 --port 9210"
metaData["pipelines-start"]="bash /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/topbds/2025/pipelines/start.sh"
metaData["TalentPlatform-LSH0577-DaemonApi"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-LSH0577-DaemonApi:app --host=0.0.0.0 --port=9000"
metaData["TalentPlatform-LSH0578-DaemonApi"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-LSH0578-DaemonApi:app --host=0.0.0.0 --port=9200"
metaData["TalentPlatform-LSH0597-DaemonApi"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-LSH0597-DaemonApi:app --host=0.0.0.0 --port=9300"
metaData["TalentPlatform-LSH0623-DaemonApi"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-LSH0623-DaemonApi:app --host=0.0.0.0 --port=9400"
metaData["TalentPlatform-LSH0627-DaemonApi"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-LSH0627-DaemonApi:app --host=0.0.0.0 --port=9030"
metaData["TalentPlatform-bdwide-DaemonApi-tripod"]="cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/TRIPOD && /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/python -m uvicorn TalentPlatform-bdwide-DaemonApi-tripod:app --host=0.0.0.0 --port=9900"
metaData["rcmdapt-ai-server"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/rcmdapt_ai_server.py"
metaData["TalentPlatform-bdwide-FileWatch"]="/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/MNTRG/TalentPlatform-bdwide-FileWatch.py"
#metaData["rclone mount"]="rclone mount 'gd-bdwide:/DATA/LABEL' '/HDD/DATA/LABEL' --daemon"

LOG_PATH=${CTX_PATH}/LOG
#LOG_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d").log
#mkdir -p ${LOG_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] LOG_PATH : $LOG_PATH"
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
#      case "$key" in
#        "TalentPlatform-LSH0605-DaemonFramework")
#		      ps -ef | grep "chrome" | awk '{print $2}' | xargs kill -9
#		      ;;
#	      *)
#		      ;;
#      esac

      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running : $key"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is not running : $key" >> ${LOG_PATH}/${LOG_NAME}
#      nohup $val >> ${LOG_PATH}/${LOG_NAME} 2>&1 &
      nohup bash -c "$val" >> ${LOG_PATH}/${LOG_NAME} 2>&1 &
  else
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: $key : ${procId}"
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Process is running with PID: $key : ${procId}" >> ${LOG_PATH}/${LOG_NAME}
  fi
done

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0

