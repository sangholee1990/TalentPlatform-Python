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
#CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/proj/indisystem/2024/SHELL
#CTX_PATH=/home/guest_user1/SYSTEMS/KIER

# 실행 파일 경로
TMP_KEY=$(mktemp .tmp-XXXXXXXXXX)
#UPD_PATH=/DATA/INPUT/INDI2024/DATA/REANALY-ERA5
UPD_PATH=/data1/REANALY-ERA5
#UPD_PATH=/only-wrf-data0/REANALY-ERA5
TMP_PATH=${UPD_PATH}/${TMP_KEY}

PY38_PATH=/SYSTEMS/anaconda3/envs/py38
#PY38_PATH=/home/guest_user1/SYSTEMS/KIER/LIB/py38

PY38_BIN=${PY38_PATH}/bin/python3

# 프로세스 종류
#MULTI_PROC=2
MULTI_PROC=5

mkdir -p $UPD_PATH
mkdir -p $TMP_PATH

find ${UPD_PATH}/.tmp* -type f -mtime +1 -exec rm -rf {} \; 2>/dev/null
find ${UPD_PATH}/.tmp* -empty -type d -mtime +1 -delete
find ${UPD_PATH} -maxdepth 1 -empty -type d -delete

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
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

  updFileName=reanaly-era5-unis_${year}${month}${day}${hour}${min}.nc
  urlFileInfo=${TMP_PATH}/${year}/${month}/${day}/${updFileName}
  mkdir -p ${urlFileInfo%/*}

cat > ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py << EOF

import cdsapi

c = cdsapi.Client(quiet = True, timeout=9999)

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
'00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'area': [
            90, -180, -90, 180,
        ],
    },
    '${urlFileInfo}')
EOF

# API키 인증
cat > $HOME/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
EOF

#  ${PY38_BIN} ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py

  ${PY38_BIN} ${TMP_PATH}/RunPython-get-reanalyEra5-unis.py &
  sleep 1s
  let cnt++

  if [ $cnt -ge ${MULTI_PROC} ]; then
    wait

    fileList=$(find ${TMP_PATH} -type f -name "*.nc" 2>/dev/null | sort -u)
    if [ ${#fileList} -le 0 ]; then continue; fi
    for fileInfo in $fileList; do
      tmpFileInfo=${fileInfo}
      updFileInfo=${fileInfo/${TMP_KEY}/}

      # 임시/업로드 파일 여부, 다운로드 용량 여부
      if [ $? -eq 0 ] && [ -e $tmpFileInfo ] && ([ ! -e ${updFileInfo} ] || [ $(stat -c %s ${tmpFileInfo}) -gt $(stat -c %s ${updFileInfo}) ]); then
          mkdir -p ${updFileInfo%/*}
          mv -f ${tmpFileInfo} ${updFileInfo}
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : mv -f ${tmpFileInfo} ${updFileInfo}"
      else
          rm -f ${tmpFileInfo}
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : rm -f ${tmpFileInfo}"
      fi
    done

    find ${UPD_PATH}/.tmp* -type f -mtime +1 -exec rm -rf {} \; 2>/dev/null
    find ${UPD_PATH}/.tmp* -empty -type d -mtime +1 -delete
    find ${UPD_PATH} -maxdepth 1 -empty -type d -delete

    cnt=0
  fi

done

# 임시 폴더 삭제
#rm -rf ${TMP_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0