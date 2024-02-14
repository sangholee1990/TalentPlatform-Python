#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# 0 */12 * * * bash /home/guest_user1/SYSTEMS/KIER/PROG/SHELL/RunShell-get-gfsncep2.sh "$(date -d "2 days ago" +\%Y-\%m-\%d\ 00:00)" "$(date +\%Y-\%m-\%d\ 00:00)"

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
UPD_PATH=/DATA/INPUT/INDI2024/DATA/SAT-SENT1
URL_PATH="https://science.globalwindatlas.info/full/sentinel1"

#PY38_PATH=/SYSTEMS/anaconda3/envs/py38
#PY38_PATH=/home/guest_user1/SYSTEMS/KIER/LIB/
#PY38_BIN=${PY38_PATH}/bin/python3

mkdir -p ${TMP_PATH}
mkdir -p ${UPD_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] TMP_PATH : $TMP_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] UPD_PATH : $UPD_PATH"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 1 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "20240214_download_list.txt"'
   echo

   exit
fi

downFile="$1"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] downFile : $downFile"

#========================================
# Run Shell
#========================================
while IFS= read -r line; do

#  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] line : $line"

  donwName=$(echo "$line" | awk -F'name=' '{print $2}' | awk -F'&' '{print $1}')
#  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] donwName : $donwName"

  # 문자열 치환을 사용하여 파일 경로 생성
  year=${donwName:8:4}
  month=${donwName:13:2}
  day=${donwName:16:2}
  hour=${donwName:19:2}
  min=${donwName:22:2}

  # URL 경로
  urlFileName=${donwName}_wind.png
  urlFileInfo=${URL_PATH}/${year}/${month}/${day}/${urlFileName}

  # 임시 경로
  tmpFileInfo=${TMP_PATH}/${urlFileName}

  # 파일 다운로드
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] URL : ${urlFileInfo}"
  wget --quiet --no-check-certificate ${urlFileInfo} -O ${tmpFileInfo}

  # 업로드 경로
  updFilePath=${UPD_PATH}/${year}/${month}/${day}
  mkdir -p ${updFilePath}

  updFileInfo=${UPD_PATH}/${year}/${month}/${day}/${urlFileName}

  # 임시/업로드 파일 여부, 다운로드 용량 여부
  if [ $? -eq 0 ] && [ -e $tmpFileInfo ] && ([ ! -e ${updFileInfo} ] || [ $(stat -c %s ${tmpFileInfo}) -gt $(stat -c %s ${updFileInfo}) ]); then
      mv -f ${tmpFileInfo} ${updFileInfo}
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : mv -f ${tmpFileInfo} ${updFileInfo}"
  else
      rm -f ${tmpFileInfo}
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CMD : rm -f ${tmpFileInfo}"
  fi
done < "$downFile"

rm -rf ${TMP_PATH}

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0