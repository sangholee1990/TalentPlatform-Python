#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
#0 */12 * * * bash /home/guest_user1/SYSTEMS/KIER/PROG/SHELL/RunShell-get-gfsncep2.sh "$(date -d "2 days ago" +\%Y-\%m-\%d\ 00:00)" "$(date +\%Y-\%m-\%d\ 00:00)"

#========================================
# Init Config
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
CTX_PATH=$(pwd)
# CTX_PATH=/home/guest_user1/SYSTEMS/KIER

# 실행 파일 경로
RUN_PATH=${CTX_PATH}/PROG/SHELL
TMP_PATH=$(mktemp -d)
UPD_PATH=/DATA/INPUT/INDI2023/DATA/GFS
#UPD_PATH=/data1/GFS
URL_PATH="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] TMP_PATH : $TMP_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] UPD_PATH : $UPD_PATH"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "2023-08-28" "2023-08-30"'
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
incDate=$srtDate
while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do

  dtYmdHm=$(date -d "${incDate}" +"%Y-%m-%d %H:%M")

  # 문자열 치환을 사용하여 파일 경로 생성
  year=$(date -d "${incDate}" +"%Y")
  month=$(date -d "${incDate}" +"%m")
  day=$(date -d "${incDate}" +"%d")
  hour=$(date -d "${incDate}" +"%H")

  incDate=$(date -d "${incDate} 6 hour")
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] dtYmdHm : $dtYmdHm"

  updFilePath=${UPD_PATH}/${year}/${month}/${day}/${hour}
  mkdir -p ${updFilePath}

#    for ftime in $(seq -w 000 001 384); do
#    for ftime in $(seq -w 000 001 049); do
  for ftime in $(seq -w 000 001 087); do
      urlFilePath=gfs.${year}${month}${day}/${hour}/atmos
      urlFileName=gfs.t${hour}z.pgrb2.0p25.f${ftime}
      tmpFileInfo=${TMP_PATH}/${urlFileName}
      updFileInfo=${updFilePath}/${urlFileName}.gb2

      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] URL : ${URL_PATH}/${urlFilePath}/${urlFileName}"
      wget --quiet --no-check-certificate ${URL_PATH}/${urlFilePath}/${urlFileName} -O ${tmpFileInfo}

      # 임시/업로드 파일 여부, 다운로드 용량 여부
      if [ $? -eq 0 ] && [ -e $tmpFileInfo ] && ([ ! -e ${updFileInfo} ] || [ $(stat -c %s ${tmpFileInfo}) -gt $(stat -c %s ${updFileInfo}) ]); then
          mv -f ${tmpFileInfo} ${updFileInfo}
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : mv -f ${tmpFileInfo} ${updFileInfo}"
      else
          rm -f ${tmpFileInfo}
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : rm -f ${tmpFileInfo}"
      fi
  done
done

rm -rf ${TMP_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0