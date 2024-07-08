#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
#CTX_PATH=$(pwd)
CTX_PATH=/SYSTEMS/PROG/SHELL/IOT
#AGENT_PATH=/root/Roverdyn/PROJ_TCP_DB/build
AGENT_PATH=/SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB/build
AGENT_NAME=SOLARMY_APP

LOG_PATH=${CTX_PATH}/LOG
LOG_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d").log

mkdir -p ${LOG_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] LOG_PATH : $LOG_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] AGENT_PATH : $AGENT_PATH"
echo

#========================================
# Run Shell
#========================================
agentProcId=$(pgrep -f "${AGENT_NAME}")

if [ -z "${agentProcId}" ]; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Agent is not running"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Agent is not running" >> ${LOG_PATH}/${LOG_NAME}
    nohup ${AGENT_PATH}/${AGENT_NAME} >> ${LOG_PATH}/${LOG_NAME} 2>&1 &
else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Agent is running with PID: ${agentProcId}"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] Agent is running with PID: ${agentProcId}" >> ${LOG_PATH}/${LOG_NAME}
fi

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0

