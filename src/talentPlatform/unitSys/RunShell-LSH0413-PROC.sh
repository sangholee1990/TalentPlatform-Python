#!/bin/bash

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Main Shell : $0"
echo

#========================================
# Set Env
#========================================
ulimit -s unlimited

# 작업 경로 설정
PY38_PATH=/usr/local/anaconda3/envs/py38/bin/python3.8
#CTX_PATH=/home/dxinyu/SYSTEMS/PROG/SHELL/CDS-API
#CTX_PATH=/SYSTEMS/PROG/SHELL/CDS-API
CTX_PATH=$(pwd)
RUN_PATH=/SYSTEMS/PROG/PYTHON/HUMAN-CNT
RUN2_PATH=/SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
VIDEO_PATH=/DATA/VIDEO
#SRC_PATH=/SYSTEMS/TMP
#INP_PATH=/DATA/INPUT/LSH0398/WFDE5
#OUT_PATH=/DATA/INPUT/LSH0398/WFDE5
#OUT_PATH=/data3/dxinyu/WFDE5


#mkdir -p $INP_PATH
#mkdir -p ${OUT_PATH}

# 다중 프로세스 개수
#MULTI_PROC=5
#MULTI_PROC=1

# 아나콘다 가상환경
#conda activate py38

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] CTX_PATH : $CTX_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] INP_PATH : $INP_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] OUT_PATH : $OUT_PATH"
#echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] MULTI_PROC : $MULTI_PROC"

#========================================
# Argument Check
#========================================
if [ "$#" -ne 2 ]; then

   echo
   echo "$# is Illegal Number of Arguments"

   echo 'Example) bash '$0' "202306/03/0242" "20230504_output.mp4"'
#   echo 'Example) bash '$0' "202305/29/1840" "20230504_output.mp4"'
   echo

   exit
fi

videoPath="$1"
videoName="$2"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] videoPath : $videoPath"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] videoName : $videoName"


#========================================
# Run Shell
#========================================
# echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] $dtYmdHm : $inpFileListCnt : $outFileListCnt : $cnt"

#echo ${PY38_PATH} ${RUN_PATH}/yolov7-object-tracking/TalentPlatform-LSH0413-detect_and_track.py --weights yolov7.pt --source "${VIDEO_PATH}/${videoPath}/${videoName}" --classes 0 --save-txt --no-trace --exist-ok --project "result" --name "${videoPath}"
#echo ${PY38_PATH} ${RUN_PATH}/YOLOv7-DeepSORT-Object-Tracking/TalentPlatform-LSH0413-deep_sort_tracking_id.py --weights yolov7.pt --source "${VIDEO_PATH}/${videoPath}/${videoName}" --classes 0 --save-txt --save-conf --exist-ok --project "result" --name "${videoPath}"

cd ${RUN2_PATH}
${PY38_PATH} ${RUN2_PATH}/TalentPlatform-LSH0413-PROC.py --videoPath ${videoPath} --videoName ${videoName} &
sleep 2s

cd ${RUN_PATH}/yolov7-object-tracking
${PY38_PATH} ${RUN_PATH}/yolov7-object-tracking/TalentPlatform-LSH0413-detect_and_track.py --weights yolov7.pt --source "${VIDEO_PATH}/${videoPath}/${videoName}" --classes 0 --save-txt --no-trace --exist-ok --project "result" --name "${videoPath}" &
sleep 2s

cd ${RUN_PATH}/YOLOv7-DeepSORT-Object-Tracking
${PY38_PATH} ${RUN_PATH}/YOLOv7-DeepSORT-Object-Tracking/TalentPlatform-LSH0413-deep_sort_tracking_id.py --weights yolov7.pt --source "${VIDEO_PATH}/${videoPath}/${videoName}" --classes 0 --save-txt --save-conf --exist-ok --project "result" --name "${videoPath}" &
sleep 2s

wait

ln -sf ${RUN_PATH}/yolov7-object-tracking/result/${videoPath} ${VIDEO_PATH}/${videoPath}/yolov7-object-tracking
ln -sf ${RUN_PATH}/YOLOv7-DeepSORT-Object-Tracking/result/${videoPath} ${VIDEO_PATH}/${videoPath}/YOLOv7-DeepSORT-Object-Tracking

cd ${RUN2_PATH}
${PY38_PATH} ${RUN2_PATH}/TalentPlatform-LSH0413-FNL.py --videoPath ${videoPath} --videoName ${videoName}

echo
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [END] Main Shell : $0"

exit 0


#cnt=0
#incDate=$srtDate
#while [ $(date -d "$incDate" +"%s") -le $(date -d "$endDate" +"%s") ]; do
#
#   #dtYmd=$(date -d "$incDate" +"%Y%m%d")
#   #dt2Ymd=$(date -d "$incDate" +"%y%m%d")
#   dtYmdHm=$(date -d "$incDate" +"%Y%m%d%H%M")
#   dtYear=$(date -d "$incDate" +"%Y")
#
#   #inpFileListCnt=$(find ${INP_PATH} -name "*_${dt2Ymd}.txt" -type f 2>/dev/null | sort -u | wc -l)
#   #outFileListCnt=$(find ${OUT_PATH}/${dtPathDate} -name "${dtYmdHm}.SZA.bin" -type f 2>/dev/null | sort -u | wc -l)
#
#   #incDate=$(date -d "${incDate} 10 minutes")
#   #incDate=$(date -d "${incDate} 1 days")
#   #incDate=$(date -d "${incDate} 1 hours")
#   #incDate=$(date -d "${incDate} 1 months")
#   incDate=$(date -d "${incDate} 1 years")
#
#   #if [ ${inpFileListCnt} -lt 3 ]; then continue; fi
#   #if [ ${outFileListCnt} -gt 0 ]; then continue; fi
#
#   echo "[$(date +"%Y-%m-%d %H:%M:%S")] [CHECK] $dtYmdHm : $inpFileListCnt : $outFileListCnt : $cnt"
#
#cat > ${SRC_PATH}/${dtYear}-Template-CDS-API.py << EOF
#import cdsapi
#
#c = cdsapi.Client()
#
#c.retrieve(
#    'derived-near-surface-meteorological-variables',
#    {
#        'version': '2.1',
#        'format': 'tgz',
#        'variable': [
#            'grid_point_altitude', 'near_surface_air_temperature', 'near_surface_specific_humidity',
#            'near_surface_wind_speed', 'rainfall_flux', 'snowfall_flux',
#            'surface_air_pressure', 'surface_downwelling_longwave_radiation', 'surface_downwelling_shortwave_radiation',
#        ],
#        'reference_dataset': 'cru',
#        'year': [
#            '${dtYear}',
#        ],
#        'month': [
#            '01', '02', '03',
#            '04', '05', '06',
#            '07', '08', '09',
#            '10', '11', '12',
#        ],
#    },
#    '${OUT_PATH}/download_${dtYear}.tar.gz')
#EOF
#
#   #/usr/local/anaconda3/envs/py38/bin/python3.8 ${CTX_PATH}/Template-CDS-API.py &
#   #/home/dxinyu/anaconda3/bin/python3.8 ${SRC_PATH}/Template-CDS-API.py
#   #/home/dxinyu/anaconda3/bin/python3.8 ${SRC_PATH}/${dtYear}-Template-CDS-API.py &
#
#   /usr/local/anaconda3/envs/py38/bin/python3.8 ${SRC_PATH}/${dtYear}-Template-CDS-API.py
#
#   mkdir -p ${OUT_PATH}/${dtYear}
#
#   tar -xvzf ${OUT_PATH}/download_${dtYear}.tar.gz -C ${OUT_PATH}/${dtYear}
#   #tar -xvzf ${OUT_PATH}/download_${dtYear}.tar.gz -C ${OUT_PATH}/${dtYear} &
#
#   sleep 2
#
#
#   let cnt++
#
#   if [ $cnt -ge ${MULTI_PROC} ]; then
#      wait
#      cnt=0
#   fi
#
#done