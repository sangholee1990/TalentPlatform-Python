#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# 0 */12 * * * bash /home/guest_user1/SYSTEMS/KIER/PROG/SHELL/RunShell-get-gfsncep2.sh "$(date -d "2 days ago" +\%Y-\%m-\%d\ 00:00)" "$(date +\%Y-\%m-\%d\ 00:00)"
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
# ps -ef | grep python3 | grep RunPython-get-reanalyEra5-unis.py | awk '{print $2}' | xargs kill -9

#========================================
# Init Config
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
CTX_PATH=$(pwd)
# CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/proj/indisystem/2024/SHELL
# CTX_PATH=/home/guest_user1/SYSTEMS/KIER

# 실행 파일 경로
TMP_PATH=$(mktemp -d)
UPD_PATH=/DATA/INPUT/INDI2024/DATA/REANALY-ERA5
#UPD_PATH=/data1/REANALY-ERA5

PY38_PATH=/SYSTEMS/anaconda3/envs/py38
#PY38_PATH=/home/guest_user1/SYSTEMS/KIER/LIB/py38

PY38_BIN=${PY38_PATH}/bin/python3

# 프로세스 종류
MULTI_PROC_CNT=5

mkdir -p $UPD_PATH

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] TMP_PATH : $TMP_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] UPD_PATH : $UPD_PATH"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

#   echo 'Example) bash '$0' "2023-08-28" "2023-08-30"'
#   echo 'Example) bash '$0' "2024-01-01 00:00" "2024-01-01 00:00"'
   echo 'Example) bash '$0' "2024-01-01 00:00" "2024-01-02 00:00"'
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
cnt=0
incDate=$srtDate
while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do

  dtYmdHm=$(date -d "${incDate}" +"%Y-%m-%d %H:%M")

  # 문자열 치환을 사용하여 파일 경로 생성
  year=$(date -d "${incDate}" +"%Y")
  month=$(date -d "${incDate}" +"%m")
  day=$(date -d "${incDate}" +"%d")
  hour=$(date -d "${incDate}" +"%H")
  min=$(date -d "${incDate}" +"%M")

  incDate=$(date -d "${incDate} 1 hour")
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] dtYmdHm : $dtYmdHm / cnt : $cnt"

#  updFilePath=${UPD_PATH}/${year}/${month}/${day}/${hour}
  updFilePath=${UPD_PATH}/${year}/${month}/${day}
  mkdir -p ${updFilePath}

  tmpFileName=reanaly-era5-unis_${year}${month}${day}${hour}${min}.nc
  tmpFileInfo=${TMP_PATH}/${tmpFileName}
  updFileInfo=${updFilePath}/${tmpFileName}

cat > ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py << EOF

import cdsapi

c = cdsapi.Client(quiet = True, timeout=10)
#c = cdsapi.Client(quiet = False, timeout=10)

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
#           'u_component_of_wind', 'v_component_of_wind',
            'all',
        ],
        'year': [
        '${year}'
        ],
        'month': [
        '${month}'
        ],
        'day': [
        '${day}'
        ],
        'time': [
        '${hour}:00'
        ],
        'area': [
            90, -180, -90, 180,
        ],
    },
    '${tmpFileInfo}')
EOF

# API키 인증
cat > $HOME/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: 38372:e61b5517-d919-47b6-93bf-f9a01ee4246f
EOF

#  ${PY38_BIN} ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py
  ${PY38_BIN} ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py &

  sleep 1s

  let cnt++

  if [ $cnt -ge ${MULTI_PROC_CNT} ]; then

    wait

    # 임시/업로드 파일 여부, 다운로드 용량 여부
    if [ $? -eq 0 ] && [ -e $tmpFileInfo ] && ([ ! -e ${updFileInfo} ] || [ $(stat -c %s ${tmpFileInfo}) -gt $(stat -c %s ${updFileInfo}) ]); then
        mv -f ${tmpFileInfo} ${updFileInfo}
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : mv -f ${tmpFileInfo} ${updFileInfo}"
    else
        rm -f ${tmpFileInfo}
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : rm -f ${tmpFileInfo}"
    fi

    cnt=0
  fi

done

# 임시 폴더 삭제
rm -rf ${TMP_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0