#!/bin/bash

#========================================
# Doc
#========================================
# cd /vol01/SYSTEMS/DMS02/PROG/PYTHON/gk2a_sst/bin
# bash RunShell-SST.sh "2025-01-01" "2026-01-01"

# 0 1 * * * bash /vol01/SYSTEMS/DMS02/PROG/PYTHON/gk2a_sst/bin/RunShell-SST.sh "$(date -d "9 days ago" +\%Y-\%m-\%d)"  "$(date +\%Y-\%m-\%d)"

#========================================
# Init Config
#========================================
ulimit -s unlimited
export LANG=en_US.UTF-8
export LC_TIME=en_US.UTF-8

# 작업 경로 설정
# /SYSTEMS/PROG/PYTHON/climate_extremes/gk2a_sst
PROJECT_DIR=/vol01/SYSTEMS/DMS02/PROG/PYTHON/gk2a_sst
SCRIPT_DIR=${PROJECT_DIR}/bin
LOG_DIR=${PROJECT_DIR}/log
LOG_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d").log
OUT_DIR=${PROJECT_DIR}/output

# 데이터 설정
PY37_BIN=/vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py37/bin/python

# 영상 설정
CTX_DIR=${PROJECT_DIR}
SRC_DIR=${PROJECT_DIR}/src

mkdir -p ${OUT_DIR}
mkdir -p ${LOG_DIR}

find ${LOG_DIR} -name "*.log" -type f -mtime +60 -delete

exec > >(tee -a "${LOG_DIR}/${LOG_NAME}") 2>&1

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

echo "[$(date +"%Y-%m-%d %H:%M:%S")] PROJECT_DIR : ${PROJECT_DIR}"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] SCRIPT_DIR : ${SCRIPT_DIR}"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] CTX_DIR : ${CTX_DIR}"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] SRC_DIR : ${SRC_DIR}"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] LOG_DIR : ${LOG_DIR}"
echo

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "2025-01-01" "2026-01-01"'
   echo

   exit
fi

srtDate="$1"
endDate="$2"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] srtDate : $srtDate"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] endDate : $endDate"

# 평년 시작일
climSrtYear=2020
climEndYear=2025

echo "[$(date +"%Y-%m-%d %H:%M:%S")] climSrtYear : ${climSrtYear}"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] climEndYear : ${climEndYear}"

#========================================
# Run Shell
#========================================
echo "[$(date +"%Y-%m-%d %H:%M:%S")] 데이터 생산중..."

#${PY37_BIN} ${PROJECT_DIR}/main.py --start_year $START_YEAR --target_year $END_YEAR --target_month 999 --target_step 999 --clim_start_year $CLIM_START_YEAR --clim_end_year $CLIM_END_YEAR
#${PY37_BIN} ${PROJECT_DIR}/past.py --clim_start_year ${CLIM_START_YEAR} --clim_end_year ${CLIM_END_YEAR} --start_year ${START_YEAR} --start_month ${START_MONTH} --end_year ${END_YEAR} --end_month ${END_MONTH}

incDate=$srtDate
while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do
    year=$(date -d "${incDate}" +"%Y")
    month=$(date -d "${incDate}" +"%m")
    day=$(date -d "${incDate}" +"%d")
    hour=00
    min=00
    runtime="${year}${month}${day}${hour}${min}"

    if [[ ! "${day}" =~ ^(01|10|20)$ ]]; then
        incDate=$(date -d "${incDate} 1 day")
        continue
    fi

    sDate=$(date -d "${incDate} - 1 month" +"%Y%m%d")
    eDate=$(date -d "${incDate} - 1 day" +"%Y%m%d")

    echo [$(date +"%Y-%m-%d %H:%M:%S")] ${PY37_BIN} ${PROJECT_DIR}/sst_main.py --sDate "${sDate}" --eDate "${eDate}" --runtime "${runtime}" --climSrtYear "${climSrtYear}" --climEndYear "${climEndYear}"
    ${PY37_BIN} ${SRC_DIR}/sst_main.py --sDate "${sDate}" --eDate "${eDate}" --runtime "${runtime}" --climSrtYear "${climSrtYear}" --climEndYear "${climEndYear}"

    incDate=$(date -d "${incDate} 1 day")
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] 데이터 생산완료"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] 영상 생산중..."

declare -A metaData
#metaData["sst01m-clim,ea"]="plot_sst_anomaly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ea020lc_%Y%m%d%H%M.png"
#metaData["sst01m-clim,ke"]="plot_sst_anomaly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ke020lc_%Y%m%d%H%M.nc"
#metaData["sst01m-clim,ko"]="plot_sst_anomaly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-clim_ko020lc_%Y%m%d%H%M.png"
metaData["sst01m-anom,ea"]="plot_sst_anomaly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ea020lc_%Y%m%d%H%M.png"
metaData["sst01m-anom,ke"]="plot_sst_anomaly_ke.py|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ke020lc_%Y%m%d%H%M.png"
metaData["sst01m-anom,ko"]="plot_sst_anomaly_ko.py|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-anom_ko020lc_%Y%m%d%H%M.png"
metaData["sst01m-diff,ea"]="plot_sst_anomaly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ea020lc_%Y%m%d%H%M.png"
metaData["sst01m-diff,ke"]="plot_sst_anomaly_ke.py|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ke020lc_%Y%m%d%H%M.png"
metaData["sst01m-diff,ko"]="plot_sst_anomaly_ko.py|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m-diff_ko020lc_%Y%m%d%H%M.png"
metaData["sst01m,ea"]="plot_sst_monthly_ea.py|${OUT_DIR}/gk2a_ami_le3_sst01m_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m_ea020lc_%Y%m%d%H%M.png"
metaData["sst01m,ke"]="plot_sst_monthly_ke.py|${OUT_DIR}/gk2a_ami_le3_sst01m_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m_ke020lc_%Y%m%d%H%M.png"
metaData["sst01m,ko"]="plot_sst_monthly_ko.py|${OUT_DIR}/gk2a_ami_le3_sst01m_ea020lc_%Y%m%d%H%M.nc|${OUT_DIR}/gk2a_ami_le3_sst01m_ko020lc_%Y%m%d%H%M.png"

varList=("sst01m" "sst01m-clim" "sst01m-anom" "sst01m-diff")
areaList=("ea" "ko" "ke")

incDate=$srtDate
while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do
    year=$(date -d "${incDate}" +"%Y")
    month=$(date -d "${incDate}" +"%m")
    day=$(date -d "${incDate}" +"%d")
    hour=00
    min=00
    runtime="${year}${month}${day}${hour}${min}"
    # echo "[$(date +"%Y-%m-%d %H:%M:%S")] runtime : ${runtime}"

    for varInfo in "${varList[@]}"; do
        for areaInfo in "${areaList[@]}"; do
            metaItem=${metaData[$varInfo,$areaInfo]}
            if [ -z "${metaItem}" ]; then continue; fi
            IFS='|' read -r src metaInfo metaInfo2 <<< "${metaItem}"

            # 데이터
            fileInfo=$(echo "${metaInfo}" | sed -e "s/%Y/$year/g" -e "s/%m/$month/g" -e "s/%d/$day/g" -e "s/%H/$hour/g" -e "s/%M/$min/g")
            filePath=${fileInfo%/*}
            fileName="${fileInfo##*/}"

            fileListCnt=$(find ${filePath} -name "${fileName}" -type f 2>/dev/null | sort -u | wc -l)
            if [ ${fileListCnt} -lt 1 ]; then continue; fi

            # 영상
            imgInfo=$(echo "${metaInfo2}" | sed -e "s/%Y/$year/g" -e "s/%m/$month/g" -e "s/%d/$day/g" -e "s/%H/$hour/g" -e "s/%M/$min/g")
            imgPath=${imgInfo%/*}
            imgName="${imgInfo##*/}"

            imgListCnt=$(find ${imgPath} -name "${imgName}" -type f 2>/dev/null | sort -u | wc -l)
            if [ ${imgListCnt} -ge 1 ]; then continue; fi

            if [ "${varInfo}" == "sst01m-anom" ]; then
              compMonth=$(date -d "${incDate}" +"%Y.%m")
              baseMonth="Climatology"
            elif [ "${varInfo}" == "sst01m-diff" ]; then
              compMonth=$(date -d "${incDate}" +"%Y.%m")
              baseMonth=$(date -d "${incDate} - 1 year" +"%Y.%m")
            elif [ "${varInfo}" == "sst01m" ]; then
              compMonth=$(date -d "${incDate}" +"%Y.%m")
              baseMonth=''
            fi
            echo [$(date +"%Y-%m-%d %H:%M:%S")] ${PY37_BIN} ${SRC_DIR}/${src} --CTX_DIR "${CTX_DIR}" --NC_DIR "${fileName}" --PNG_DIR "${imgName}" --COMP_MONTH "${compMonth}" --BASE_MONTH "${baseMonth}" --VAR_INFO "${varInfo^^}"
            ${PY37_BIN} ${SRC_DIR}/${src} --CTX_DIR "${CTX_DIR}" --NC_DIR "${fileName}" --PNG_DIR "${imgName}" --COMP_MONTH "${compMonth}" --BASE_MONTH "${baseMonth}" --VAR_INFO "${varInfo^^}"
        done
    done

    incDate=$(date -d "${incDate} 1 day")
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] 영상 생산완료"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0