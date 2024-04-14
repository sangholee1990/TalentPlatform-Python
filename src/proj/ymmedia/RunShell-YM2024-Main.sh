#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# DOC
#========================================
# :set paste

#========================================
# Init Config
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
CTX_PATH=$(pwd)
#CTX_PATH=/home/guest_user1/SYSTEMS/KIER
#CTX_PATH=/wind_home/jinyoung/SYSTEMS/KIER

# 아나콘다 가상환경 활성화
#PY38_PATH=/SYSTEMS/anaconda3/envs/py38
#PY38_PATH=/home/guest_user1/SYSTEMS/KIER/LIB/py38
#PY38_PATH=/wind_home/jinyoung/SYSTEMS/KIER/LIB/py38

#PY38_BIN=${PY38_PATH}/bin/python3
#PY38_BIN=${PY38_PATH}/bin/python3

# 실행경로 설정
#RUN_PATH=${CTX_PATH}/PROG/PYTHON/extract

# 작업경로 이동
#cd $RUN_PATH

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
   echo 'Example) bash '$0' "1981-01-01" "1981-02-01"'
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
metaData["ECMWF","ALL"]="/data03/climate_change/ecmwf/era5/%Y/%m/%d/*.grib"

modelList=("ECMWF")
keyList=("ALL")

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
	incDate=$(date -d "${incDate} 1 years")
#	incDate=$(date -d "${incDate} 1 hour")
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

#		  fileListCnt=$(find ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u | wc -l)
      fileListCnt=$(find -L ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u | wc -l)
		  if [ ${fileListCnt} -lt 1 ]; then continue; fi

#		  fileList=$(find ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u)
		  fileList=$(find -L ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u)
		  for fileInfo in $fileList; do
			  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] fileInfo : $fileInfo"
			  ncl_quicklook -v "2T_GDS0_SFC" "$fileInfo" | egrep "Minimum Maximum"
#			  break
		  done
	done
	done

done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0
