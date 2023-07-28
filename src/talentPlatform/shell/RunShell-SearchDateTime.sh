#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
CTX_PATH=/SYSTEMS/PROG/SHELL/CDS-API
INP_PATH=/DATA/INPUT/LSH0398
OUT_PATH=/DATA/OUTPUT/LSH0398

mkdir -p $INP_PATH
mkdir -p $OUT_PATH

# 다중 프로세스 개수
MULTI_PROC=5
#MULTI_PROC=1

# 아나콘다 가상환경
#conda activate py38

#cat > $HOME/.cdsapirc << EOF
#url: https://cds.climate.copernicus.eu/api/v2
#key: 38372:e61b5517-d919-47b6-93bf-f9a01ee4246f
#EOF

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] INP_PATH : $INP_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] OUT_PATH : $OUT_PATH"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] MULTI_PROC : $MULTI_PROC"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   #echo 'Example) bash '$0' "2022-03-04 23:30" "2021-03-04 23:40"'
   #echo 'Example) bash '$0' "2020-03-01" "2022-12-01"'
   echo 'Example) bash '$0' "1979-01-01" "2020-01-01"'
#   echo 'Example) bash '$0' "19790101" "20200101"'
   #echo 'Example) bash '$0' "2022-06-01" "2022-06-02"'
   #echo 'Example) bash '$0' "2020301" "20200331"'
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

   #dtYmd=$(date -d "$incDate" +"%Y%m%d")
   #dt2Ymd=$(date -d "$incDate" +"%y%m%d")
   dtYmdHm=$(date -d "$incDate" +"%Y%m%d%H%M")
   dtYear=$(date -d "$incDate" +"%Y")

   #inpFileListCnt=$(find ${INP_PATH} -name "*_${dt2Ymd}.txt" -type f 2>/dev/null | sort -u | wc -l)
   #outFileListCnt=$(find ${OUT_PATH}/${dtPathDate} -name "${dtYmdHm}.SZA.bin" -type f 2>/dev/null | sort -u | wc -l)

   #incDate=$(date -d "${incDate} 10 minutes")
   #incDate=$(date -d "${incDate} 1 days")
   #incDate=$(date -d "${incDate} 1 hours")
   #incDate=$(date -d "${incDate} 1 months")
   incDate=$(date -d "${incDate} 1 years")

   #if [ ${inpFileListCnt} -lt 3 ]; then continue; fi
   #if [ ${outFileListCnt} -gt 0 ]; then continue; fi

   echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] $dtYmdHm : $inpFileListCnt : $outFileListCnt : $cnt"

#   /usr/local/anaconda3/envs/py38/bin/python3.8 ${CTX_PATH}/Template-CDS-API.py &
#   tar -xvzf ${OUT_PATH}/download_202001.tar.gz -C ${OUT_PATH}

   let cnt++

   if [ $cnt -ge ${MULTI_PROC} ]; then
      wait
      cnt=0
   fi

done

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"
