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
CTX_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/ncl/REANALY-ERA5

# 아나콘다 가상환경 활성화
#PY38_BIN=${CTX_PATH}/LIB/py38/bin/python3

# NCL 실행환경
NCL_BIN=/usr/local/ncl-6.6.2/bin/ncl

# 작업경로 설정
#RUN_PATH=${CTX_PATH}/PROG/PYTHON/extract
RUN_PATH=${CTX_PATH}

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
metaData["REANALY-ERA5","ALL"]="/DATA/INPUT/LSH0545/ecmwf_mean_daily_2t_out_%m%d.csv"

modelList=("REANALY-ERA5")
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
#	incDate=$(date -d "${incDate} 1 hour")
	incDate=$(date -d "${incDate} 1 day")
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
		    fileName=$(basename $fileInfo)
		    mainTitle=${fileName/.csv/}
		    saveImg=/DATA/FIG/LSH0545/${fileName/.csv/.png}
        mkdir -p ${saveImg%/*}

			  echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] fileInfo : $fileInfo"
			  ${NCL_BIN} -Q 'fileInfo="'$fileInfo'"' 'mainTitle="'$mainTitle'"' 'saveImg="'$saveImg'"' ${RUN_PATH}/RunNcl-reanalyEra5-vis.ncl

#			  break
		  done
	done
	done

done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0
