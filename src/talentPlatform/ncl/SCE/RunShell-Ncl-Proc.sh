#!/bin/bash

# bash RunShell-Ncl-Proc.sh 20000101 20200101

function globalVar {

   ulimit -s unlimited

   CTX_PATH=$(pwd)
   echo "[CHECK] CTX_PATH : $CTX_PATH"

   TMP_PATH=${CTX_PATH}/TMP
   mkdir -p $TMP_PATH
   echo "[CHECK] TMP_PATH : $TMP_PATH"

   CSV_PATH=${CTX_PATH}/CSV
   mkdir -p $CSV_PATH
   echo "[CHECK] CSV_PATH : $CSV_PATH"
   # rm -rf $CSV_PATH

  TEMPLATE_PATH=${CTX_PATH}/TEMPLATE
  mkdir -p $TEMPLATE_PATH
  echo "[CHECK] TEMPLATE_PATH : $TEMPLATE_PATH"

  RES_PATH=${CTX_PATH}/RESULT
  mkdir -p $RES_PATH
  echo "[CHECK] RES_PATH : $RES_PATH"


}

function argsNumberValid() {

  local argsNumber=$#

  if [ $argsNumber -ne 2 ]; then

    echo
    echo "$argsNumber is Illegal Number of Arguments"

    echo "Example) RunShell-Ncl-Write.sh %Y%m%d%H%M %Y%m%d%H%M"
    echo "         RunShell-Ncl-Write.sh 202009130021 202009130455"

    exit 1
  fi

  return 0
}

function parseDateTime {

  local sDateTime="$1"
  local sDate=${sDateTime:0:8}
  local sHour=${sDateTime:8:2}
  local sMin=${sDateTime:10:2}
  local sSec=${sDateTime:12:2}

  if [ "${sHour}" == "" ]; then sHour="00"; fi
  if [ "${sMin}" == "" ]; then sMin="00"; fi
  if [ "${sSec}" == "" ]; then sSec="00"; fi

  dtDateTime=$(date -d "${sDate} ${sHour}:${sMin}:${sSec}" +"%Y-%m-%d %H:%M:%S")

  dtUnix=$(date -d "${dtDateTime}" +"%s")
  dtYmdHmsFmt=$(date -d "${dtDateTime}" +"%Y-%m-%d %H:%M:%S")
  dtYmdHms=$(date -d "${dtDateTime}" +"%Y%m%d%H%M%S")
  dtYmdHmFmt=$(date -d "${dtDateTime}" +"%Y-%m-%d %H:%M")
  dtYmdHm=$(date -d "${dtDateTime}" +"%Y%m%d%H%M")
  dtYm=$(date -d "${dtDateTime}" +"%Y%m")
  dtYmd=$(date -d "${dtDateTime}" +"%Y%m%d")

  dtYear=$(date -d "${dtDateTime}" +"%Y")
  dtMonth=$(date -d "${dtDateTime}" +"%m")
}

function parseFileInfo {

  local fileInfo="$1"

  fileDir=${fileInfo%/*}
  fileName="${fileInfo##*/}"
}

#========================================
# Argument Check
#========================================
reqSrtDateTime="$1"
reqEndDateTime="$2"

echo "[CHECK] args : $reqSrtDateTime"
echo "[CHECK] args2 : $reqEndDateTime"

argsNumberValid "$@"

#========================================
# Set Env
#========================================
globalVar

#========================================
# Run Shell
#========================================
echo "[START] Main Shell : " $0

parseDateTime $reqSrtDateTime
srtDateTimeUnix=$dtUnix
srtDateTimeFmt=$dtYmdHmFmt

incDateTimeUnix=$dtUnix

parseDateTime $reqEndDateTime
endDateTimeUnix=$dtUnix
endDateTimeFmt=$dtYmdHmsFmt

echo "[CHECK] srtDateTimeFmt: $srtDateTimeFmt"
echo "[CHECK] endDateTimeFmt : $endDateTimeFmt"

while [ $incDateTimeUnix -le $endDateTimeUnix ]; do

  incDateTime=$(date +"%Y%m%d%H%M%S" -d @$incDateTimeUnix)

  parseDateTime $incDateTime
  dtDateTimeFmt=$dtYmdHmFmt
  dtYear=$dtYear
  dtMonth=$dtMonth

  echo "[CHECK] dtDateTimeFmt : $dtDateTimeFmt"

  #  incDateTimeUnix=$(date +"%s" -d "${dtDateTimeFmt} 1 min")
  incDateTimeUnix=$(date +"%s" -d "${dtDateTimeFmt} 1 month")

  inFile="${CTX_PATH}/SCE_monthly_nhsce_v01r01_19661004_20211004.nc"
#  echo "[CHECK] inFile : $inFile"

  saveFile="${CSV_PATH}/SCE_Point_Data_60N_90E_${dtYear}_${dtMonth}.csv"
  echo "[CHECK] saveFile : $saveFile"

  cat ${TEMPLATE_PATH}/TEMPLATE_sce_anomaly_map_write_output.ncl |
    sed -e "s:%preYear:$dtYear:g" \
      -e "s:%preMonth:$dtMonth:g" \
      -e "s:%inFile:$inFile:g" \
      -e "s:%saveFile:$saveFile:g" > ${TMP_PATH}/RUN_sce_anomaly_map_write_output.ncl

  # ncl ${TMP_PATH}/RUN_sce_anomaly_map_write_output.ncl  > /dev/null 2>&1 &
  ncl ${TMP_PATH}/RUN_sce_anomaly_map_write_output.ncl


done

resFile="${RES_PATH}/SCE_Point_Data_60N_90E.csv"
echo "[CHECK] resFile : resFile"
find ${CSV_PATH} -type f -name "SCE_Point_Data_60N_90E_*_*.csv"  2>/dev/null | sort -u | xargs cat > ${resFile}

echo "[END] Main Shell : " $0
