#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell
# nohup bash /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/shell/RunShell-LSH0547.sh &
# tail -f nohup.out

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
PY_PATH=/SYSTEMS/anaconda3/envs/py38-test/bin/python3.8

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
# GW
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1981-2010" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1981-2010" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1981-2010" --selIdx "2" &

${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1990-2020" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1990-2020" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "1990-2020" --selIdx "2" &

${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "2010-2020" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "2010-2020" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GW-proc.py --analyList "2010-2020" --selIdx "2" &

# GL
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1981-2010" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1981-2010" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1981-2010" --selIdx "2" &

${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1990-2020" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1990-2020" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "1990-2020" --selIdx "2" &

${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "2010-2020" --selIdx "0" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "2010-2020" --selIdx "1" &
${PY_PATH} ${RUN_PATH}/TalentPlatform-LSH0547-GL-proc.py --analyList "2010-2020" --selIdx "2" &

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0