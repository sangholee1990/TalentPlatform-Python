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
#CTX_PATH=$(pwd)
# KIM, LDAPS/RDAPS
CTX_PATH=/home/guest_user1/SYSTEMS/KIER

# 아나콘다 가상환경 활성화
#source /home/indisystem/.bashrc
#conda activate py38
#PY38_BIN=/home/indisystem/.conda/envs/py38/bin/python3
#PY38_BIN=/vol01/SYSTEMS/KIER/LIB/py38/bin/python3
#PY38_BIN=/vol01/SYSTEMS/KIER/LIB/py38/bin/python3
PY38_BIN=${CTX_PATH}/LIB/py38/bin/python3

# 작업경로 설정
RUN_PATH=${CTX_PATH}/PROG/PYTHON/extract

# 작업경로 이동
cd $RUN_PATH

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] RUN_PATH : $RUN_PATH"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

#   echo 'Example) bash '$0' "2023-06-27 00:00" "2023-07-02 00:00"'
#   echo 'Example) bash '$0' "2023-08-01 00:00" "2023-08-01 00:00"'
   echo 'Example) bash '$0' "2010-01-01 00:00" "2010-12-31 00:00"'
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
declare -A metaData

# 개발 서버
#metaData["KIER-LDAPS","UNIS"]="/vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.%Y-%m-%d_%H:*:*.nc"
#metaData["KIER-LDAPS","PRES"]="/vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_%Y-%m-%d_%H:*:*.nc"
#metaData["KIER-RDAPS","UNIS"]="/vol01/DATA/MODEL/KIER-RDAPS/wrfsolar_d02.%Y-%m-%d_%H:*:*.nc"
#metaData["KIER-RDAPS","PRES"]="/vol01/DATA/MODEL/KIER-RDAPS/wrfout_d02_%Y-%m-%d_%H:*:*.nc"
#metaData["KIM","UNIS"]="/vol01/DATA/MODEL/KIM/r030_v040_ne36_unis_h*.%Y%m%d%H.gb2"
#metaData["KIM","PRES"]="/vol01/DATA/MODEL/KIM/r030_v040_ne36_pres_h*.%Y%m%d%H.gb2"
#metaData["LDAPS","UNIS"]="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_unis_h*.%Y%m%d%H.gb2"
#metaData["LDAPS","PRES"]="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_pres_h*.%Y%m%d%H.gb2"
#metaData["RDAPS","UNIS"]="/vol01/DATA/MODEL/RDAPS/g120_v070_erea_unis_h*.%Y%m%d%H.gb2"
#metaData["RDAPS","PRES"]="/vol01/DATA/MODEL/RDAPS/g120_v070_erea_pres_h*.%Y%m%d%H.gb2"

# 운영서버
metaData["KIER-LDAPS-2K","UNIS"]="/data1/LDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"
metaData["KIER-LDAPS-2K","PRES"]="/data1/LDAPS-WRF/%Y/%m/%d/%H/wrfout_d02*"
metaData["KIER-LDAPS-2K-30M","UNIS"]="/data1/LDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"
metaData["KIER-LDAPS-2K-60M","UNIS"]="/data1/LDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"

metaData["KIER-RDAPS-3K","UNIS"]="/data1/RDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"
metaData["KIER-RDAPS-3K","PRES"]="/data1/RDAPS-WRF/%Y/%m/%d/%H/wrfout_d02*"
metaData["KIER-RDAPS-3K-30M","UNIS"]="/data1/RDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"
metaData["KIER-RDAPS-3K-60M","UNIS"]="/data1/RDAPS-WRF/%Y/%m/%d/%H/wrfsolar_d02*"

metaData["KIER-WIND","ALL"]="/thermal1/OUT/wrf%Y_%m/wrfout_d04s_%Y-%m-%d*"
metaData["KIER-WIND-30M","ALL"]="/thermal1/OUT/wrf%Y_%m/wrfout_d04s_%Y-%m-%d*"
metaData["KIER-WIND-60M","ALL"]="/thermal1/OUT/wrf%Y_%m/wrfout_d04s_%Y-%m-%d*"

metaData["KIM-3K","UNIS"]="/data1/%Y/%m/%d/%H/r030_v040_ne36_unis_h*.%Y%m%d%H.gb2"
metaData["KIM-3K","PRES"]="/data1/%Y/%m/%d/%H/r030_v040_ne36_pres_h*.%Y%m%d%H.gb2"

metaData["LDAPS-1.5K","UNIS"]="/only-wrf-data2/Forecast/LDAPS/%Y/%m/%d/%H/l015_v070_erlo_unis_h*.%Y%m%d%H.gb2"
metaData["LDAPS-1.5K","PRES"]="/only-wrf-data2/Forecast/LDAPS/%Y/%m/%d/%H/l015_v070_erlo_pres_h*.%Y%m%d%H.gb2"

metaData["RDAPS-12K","UNIS"]="/only-wrf-data2/Forecast/RDAPS/%Y/%m/%d/%H/g120_v070_erea_unis_h*.%Y%m%d%H.gb2"
metaData["RDAPS-12K","PRES"]="/only-wrf-data2/Forecast/RDAPS/%Y/%m/%d/%H/g120_v070_erea_pres_h*.%Y%m%d%H.gb2"

#modelList=("LDAPS-1.5K")
#modelList=("RDAPS-12K")
#modelList=("KIM-3K")
#keyList=("UNIS" "PRES")

#modelList=("KIER-RDAPS-3K" "KIER-RDAPS-3K-30M" "KIER-RDAPS-3K-60M")
#modelList=("KIER-LDAPS-2K" "KIER-LDAPS-2K-30M" "KIER-LDAPS-2K-60M")
#keyList=("UNIS" "PRES")

#modelList=("KIER-WIND" "KIER-WIND-30M" "KIER-WIND-60M")
#keyList=("ALL")

incDate=$srtDate
while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do

	dtYmdHm=$(date -d "${incDate}" +"%Y-%m-%d %H:%M")
    
	# 문자열 치환을 사용하여 파일 경로 생성
  year=$(date -d "${incDate}" +"%Y")
  month=$(date -d "${incDate}" +"%m")
  day=$(date -d "${incDate}" +"%d")
  hour=$(date -d "${incDate}" +"%H")
  min=$(date -d "${incDate}" +"%M")
  sec=$(date -d "${incDate}" +"%S")

	#echo $incDate
	incDate=$(date -d "${incDate} 1 hour")
#	incDate=$(date -d "${incDate} 1 days")
#	incDate=$(date -d "${incDate} 1 months")
	echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] dtYmdHm : $dtYmdHm"

	for model in ${modelList[@]}; do
	for key in ${keyList[@]}; do
	    metaInfo=${metaData[$model,$key]}
		  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] model : $model / key : $key"
		  #echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] metaInfo : $metaInfo"

		  fileInfoPtrn=$(echo "$metaInfo" | sed -e "s/%Y/$year/g" -e "s/%m/$month/g" -e "s/%d/$day/g" -e "s/%H/$hour/g" -e "s/%M/$min/g")
		  filePath=${fileInfoPtrn%/*}
   		fileName="${fileInfoPtrn##*/}"

		  fileListCnt=$(find ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u | wc -l)
		  if [ ${fileListCnt} -lt 1 ]; then continue; fi

		  fileList=$(find ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u)
		  for fileInfo in $fileList; do
			  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] fileInfo : $fileInfo"
			  ${PY38_BIN} ${RUN_PATH}/main.py "$fileInfo" "$model"
#			  break
		  done
	done
	done

done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0
