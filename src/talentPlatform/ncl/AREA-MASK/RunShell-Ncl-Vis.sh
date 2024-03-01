#!/bin/bash

CTX_PATH=$(pwd)

uid=$(date +"%Y%m/%d/%H/%M")

#DATA_PATH=/DATA/INPUT/LSH0360
DATA_PATH=/DATA/INPUT/LSH0495
FIG_PATH=${CTX_PATH}/FIG
CFG_PATH=${CTX_PATH}/CONFIG
TMP_PATH=${CTX_PATH}/TMP

mkdir -p ${TMP_PATH}
mkdir -p ${FIG_PATH}
mkdir -p ${CFG_PATH}

parseFileInfo() {

  local fileInfo="$1"

  fileDir=${fileInfo%/*}
  fileName="${fileInfo##*/}"
}


key="$1"
echo "[CHECK] key : $key"

#fileList=$(find ${DATA_PATH} -type f -name "${key}*.nc" | sort -u)
#fileList=$(find ${DATA_PATH} -type f -name "CarbonMonitor-org_total_20190101.nc" | sort -u)
fileList=$(find ${DATA_PATH} -type f -name "*.nc" | sort -u)
echo "[CHECK] fileList : $fileList"

for fileInfo in $fileList; do
   echo "[CHECK] fileInfo : $fileInfo"

   parseFileInfo $fileInfo
   fileNameNoExt=${fileName/.nc/}
   saveImg=${FIG_PATH}/${fileNameNoExt}
   cfgInfo=${CFG_PATH}/rgb.txt

# All
#cat ${CFG_PATH}/TEMPLATE.ncl |
#cat ${CFG_PATH}/TEMPLATE-CAR-MON.ncl |
cat ${CFG_PATH}/TEMPLATE-AREA.ncl |
  sed -e "s:%fileInfo:${fileInfo}:g" \
    -e "s:%cfgInfo:${cfgInfo}:g" \
    -e "s:%var:emission:g" \
    -e "s:%saveImg:${saveImg}:g" > ${TMP_PATH}/RUN.ncl

ncl ${TMP_PATH}/RUN.ncl

#convert -rotate -90 -density 1000 -transparent white ${saveImg}.ps ${saveImg}.png
#convert -rotate -90 -transparent white ${saveImg}.ps ${saveImg}.png
convert -transparent white ${saveImg}.png ${saveImg}.png


# AREA MASK
minLonList=(-14.00478 -23.08840 -85.35378 -124.74968 64.35419)
maxLonList=(44.71888 51.44285 -33.94776 -63.57780 145.21390)
minLatList=(34.430346 -34.41006 -56.24722 24.55069 4.38144)
maxLatList=(60.090 37.61990 14.72564  50.05260 55.02741)
nameList=(EU AF AM US AS)

for i in ${!nameList[@]}; do
   echo $i ${nameList[i]} ${minLonList[i]}

   saveImg=${FIG_PATH}/${fileNameNoExt}-${nameList[i]}

cat ${CFG_PATH}/TEMPLATE-AREA-MASK.ncl |
  sed -e "s:%fileInfo:${fileInfo}:g" \
    -e "s:%cfgInfo:${cfgInfo}:g" \
    -e "s:%minLon:${minLonList[i]}:g" \
    -e "s:%maxLon:${maxLonList[i]}:g" \
    -e "s:%minLat:${minLatList[i]}:g" \
    -e "s:%maxLat:${maxLatList[i]}:g" \
    -e "s:%var:emission:g" \
    -e "s:%saveImg:${saveImg}:g" > ${TMP_PATH}/RUN.ncl

ncl ${TMP_PATH}/RUN.ncl

convert -transparent white ${saveImg}.png ${saveImg}.png
done

done

#wait
#rm -f ${FIG_PATH}/*.ps
