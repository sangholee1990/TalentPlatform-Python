#!/bin/bash

CTX_PATH=$(pwd)

uid=$(date +"%Y%m/%d/%H/%M")

#DATA_PATH=${CTX_PATH}/DATA
#DATA_PATH=/DATA/OUTPUT/LSH0344
#DATA_PATH=/DATA/OUTPUT/LSH0418/EMI
#DATA_PATH=/DATA/INPUT/LSH0457
DATA_PATH=/DATA/INPUT/LSH0495
FIG_PATH=${CTX_PATH}/FIG
CFG_PATH=${CTX_PATH}/CONFIG
#RUN_PATH=/tmp/${uid}
RUN_PATH=${CTX_PATH}/RUN/${uid}

mkdir -p ${RUN_PATH}
mkdir -p ${FIG_PATH}
mkdir -p ${CFG_PATH}

parseFileInfo() {

  local fileInfo="$1"

  fileDir=${fileInfo%/*}
  fileName="${fileInfo##*/}"
}


key="$1"
echo "[CHECK] key : $key"

fileList=$(find ${DATA_PATH} -type f -name "${key}*.nc" | sort -u)
echo "[CHECK] fileList : $fileList"

for fileInfo in $fileList; do

   echo "[CHECK] fileInfo : $fileInfo"

   parseFileInfo $fileInfo
   fileNameNoExt=${fileName/.nc/}
   saveImg=${FIG_PATH}/${fileNameNoExt}
   cfgInfo=${CFG_PATH}/rgb.txt

   echo "[CHECK] fileNameNoExt : $fileNameNoExt"
   #type=${fileName#*_}
   #type=${type%.*}
   #echo "[CHECK] type : $type"

cat ${CFG_PATH}/TEMPLATE-EMI-DIFF.ncl |
  sed -e "s:%fileInfo:${fileInfo}:g" \
    -e "s:%cfgInfo:${cfgInfo}:g" \
    -e "s:%var:emission:g" \
    -e "s:%saveImg:${saveImg}:g" > ${RUN_PATH}/run.ncl

#    -e "s:%var:${fileNameNoExt}:g" \

ncl ${RUN_PATH}/run.ncl
#convert -rotate -90 -density 1000 -transparent white ${saveImg}.ps ${saveImg}.png &
#convert -transparent white ${saveImg}.png ${saveImg}.png
convert -white ${saveImg}.png ${saveImg}.png

   echo "[CHECK] saveImg : ${saveImg}.png"
done

#wait
#rm -f ${FIG_PATH}/*.ps
