#!/bin/bash

# 쉘종류, 메인쉘, 시작날짜, 종료날짜 (연도별 기간 검색), 시자
# bash RunShell-Ncl-Vis.sh 20000101 20210101 3

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

  FIG_PATH=${CTX_PATH}/FIG
  mkdir -p $FIG_PATH
  echo "[CHECK] FIG_PATH : $FIG_PATH"

}

function argsNumberValid() {

  local argsNumber=$#

  if [ $argsNumber -ne 3 ]; then

    echo
    echo "$argsNumber is Illegal Number of Arguments"

    echo "Example) RunShell-Ncl-Vis.sh %Y%m%d%H%M %Y%m%d%H%M %m %m"
    echo "         RunShell-Ncl-Vis.sh 20201101 20201201 2"
    echo "         RunShell-Ncl-Vis.sh 20200201 20200301 2"
    echo "         RunShell-Ncl-Vis.sh 20200501 20200501 2"
    echo "         RunShell-Ncl-Vis.sh 20200701 20200701 2"
    echo "         RunShell-Ncl-Vis.sh 20170101 20210101 12"

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
invMonth="$3"

echo "[CHECK] args : $reqSrtDateTime"
echo "[CHECK] args2 : $reqEndDateTime"
echo "[CHECK] args3 : $invMonth"

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
echo "[CHECK] invMonth : $invMonth"

while [ $incDateTimeUnix -le $endDateTimeUnix ]; do

  incDateTime=$(date +"%Y%m%d%H%M%S" -d @$incDateTimeUnix)

  parseDateTime $incDateTime
  dtDateTimeFmt=$dtYmdHmFmt
  dtYmd=$dtYmd
  dtYear=$dtYear

  preDateTime=$(seq 0 1 $invMonth | xargs -I {} date -d "${dtYmd} {} month" +"%Y%m%d%H%M" | sort -u)

  cnt=0
  strInfo=""
  for preDateTime in $preDateTime; do
    echo "[CHECK] preDateTime : $preDateTime"
    parseDateTime $preDateTime
    preYear=$dtYear
    preMonth=$dtMonth

    if [ $cnt -eq 0 ]; then
      strTmp="( (year .eq. $preYear) .and. (month .eq. $preMonth) )"
      dtSrtYm=${dtYear}${dtMonth}
    else
      strTmp=".or. ( (year .eq. $preYear) .and. (month .eq. $preMonth) )"
      dtEndYm=${dtYear}${dtMonth}
    fi

    strInfo+="${strTmp} "

    let cnt++
  done

  echo "[CHECK] strInfo : $strInfo"
  echo "[CHECK] dtDateTimeFmt (dtSrtYm - dtEndYm) : $dtDateTimeFmt ($dtSrtYm - $dtEndYm)"

  # incDateTimeUnix=$(date +"%s" -d "${dtDateTimeFmt} 1 min")
  # incDateTimeUnix=$(date +"%s" -d "${dtDateTimeFmt} 3 month")
  incDateTimeUnix=$(date +"%s" -d "${dtDateTimeFmt} 1 Year")

  inFile="${CTX_PATH}/SCE_monthly_nhsce_v01r01_19661004_20211004.nc"
  # echo "[CHECK] inFile : $inFile"

  saveImg="${FIG_PATH}/sce_anomaly_${dtSrtYm}_${dtEndYm}_${invMonth}"
  echo "[CHECK] saveImg : $saveImg"

  #  cat ${TEMPLATE_PATH}/TEMPLATE_sce_anomaly_map_only.ncl |
  #    sed -e "s:%preYear:$dtYear:g" \
  #      -e "s:%preSrtMonth:$srtMonth:g" \
  #      -e "s:%preEndMonth:$endMonth:g" \
  #      -e "s:%inFile:$inFile:g" \
  #      -e "s:%figname:$saveImg:g" > ${TMP_PATH}/RUN_sce_anomaly_map_only.ncl

  cat ${TEMPLATE_PATH}/TEMPLATE_sce_anomaly_map_only.ncl |
    sed -e "s:%dtSrtYm:$dtSrtYm:g" \
      -e "s:%dtEndYm:$dtEndYm:g" \
      -e "s:%timeIdxPre:$strInfo:g" \
      -e "s:%inFile:$inFile:g" \
      -e "s:%figname:$saveImg:g" >${TMP_PATH}/RUN_sce_anomaly_map_only.ncl

  # ncl ${TMP_PATH}/RUN_sce_anomaly_map_only.ncl  > /dev/null 2>&1 &
  ncl ${TMP_PATH}/RUN_sce_anomaly_map_only.ncl

done

echo "[END] Main Shell : " $0
