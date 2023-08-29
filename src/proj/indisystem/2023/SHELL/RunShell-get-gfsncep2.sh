#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

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
RUN_PATH=${CTX_PATH}/PROG/PYTHON/extract
TMP_PATH=/tmp
UPD_PATH=/DATA/INPUT/INDI2023/DATA/GFS
#UPD_PATH=/data1/GFS
URL_PATH="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

#========================================
# Run Shell
#========================================
#srtDate=$(date +%C%y%m%d%H -d -16 hour)
srtDate=$(date -d "16 hour ago" +"%C%y%m%d%H")
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] srtDate : $srtDate"

year=$(echo $srtDate | cut -c01-04)
month=$(echo $srtDate | cut -c05-06)
day=$(echo $srtDate | cut -c07-08)

for runcycle in $(seq -w 0 6 18); do
    updFilePath=${UPD_PATH}/${year}/${month}/${day}/${runcycle}
    mkdir -p ${updFilePath}

#    for ftime in $(seq -w 0 1 384); do
#    for ftime in $(seq -w 0 1 49); do
    for ftime in $(seq -w 0 1 87); do
        urlFilePath=gfs.${year}${month}${day}/${runcycle}/atmos
        urlFileName=gfs.t${runcycle}z.pgrb2.0p25.f${ftime}
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

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0