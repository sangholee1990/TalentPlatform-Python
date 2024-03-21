#!/bin/bash

#source ~/.bashrc
#conda activate py38

# contextPath="/home/fedora/PROG/ShellScript/Youtube-Upload"
#contextPath="/SYSTEMS/PROG/ShellScript/Youtube-Upload"
contextPath="/SYSTEMS/PROG/SHELL/VIDEO/YOUTUBE-UPLOAD"

# srtDateTime="$1"
# endDateTime="$2"
srtDateTime="$(date +'%Y-%m-%d %H:%M' -d '1 days ago')"
#srtDateTime="$(date +'%Y-%m-%d %H:%M' -d '2 days ago')"
endDateTime="$(date +'%Y-%m-%d %H:%M')"


# echo $1 >> crontab.log
# echo $2 >> crontab.log

echo "[CHECK] srtDateTime : $srtDateTime"
echo "[CHECK] endDateTime : $endDateTime"

dtSrtUnix=$(date -d "${srtDateTime}" +%s)
dtEndUnix=$(date -d "${endDateTime}" +%s)
dtIncUnix=$dtSrtUnix

while [ $dtIncUnix -le $dtEndUnix ]; do
  dtYmdHm=$(date +"%Y-%m-%d %H:%M" -d @$dtIncUnix)
  dtYmd=$(date +"%Y%m%d" -d @$dtIncUnix)

  dtIncUnix=$(date +"%s" -d "${dtYmdHm} 1 days")

  fileList=$(find /DATA/OUTPUT/VIDEO -name "*.png_*_${dtYmd}*-*_KST.mp4")
  # fileList=$(find /SYSTEMS/OUTPUT/VIDEO -name "*.png_*_${dtYmd}*-*_*.mp4" | grep "ko")

  # echo $fileList
  if [ ${#fileList} -lt 1 ]; then continue; fi
  
  for fileInfo in $fileList; do
    fileName="${fileInfo##*/}"

    tmpDateTime=$(echo $fileName | awk -F '[_]' '{print $11}')

    dataYmd=$(date +"%Y%m%d" -d "${tmpDateTime:0:8} ${tmpDateTime:8:4} 9 hours")
    dataPeriod=$(date +"%B %d, %Y" -d "${tmpDateTime:0:8} ${tmpDateTime:8:4} 9 hours")

    type=$(echo $fileName | awk -F '[_]' '{print $7}')
    case $type in
    "ir105")
      dataType="Infrared Radiation 10.5 um"
      ;;
    "rgb-daynight")
      dataType="RGB Day/Night Composite"
      ;;
    "rgb-true")
      dataType="RGB True"
      ;;
    "rgb-airmass")
      dataType="RGB Airmass"
      ;;
    "rgb-ash")
      dataType="RGB Ash"
      ;;
    "rgb-dust")
      dataType="RGB Dust"
      ;;
    esac

    area=$(echo $fileName | awk -F '[_]' '{print $8}' | cut -c 1-2)
    case $area in
    "fd")
      dataArea="Full Disk"
      ;;
    "ea")
      dataArea="East Asia"
      ;;
    "ko")
      dataArea="Korea Peninsula"
      ;;
    esac

    title="[Daily] ${dataPeriod} GK2A Satellite ${dataType} | ${dataArea} | ${dataYmd}"

    echo "[$dtYmdHm] $title : $fileInfo"

    cat ${contextPath}/Template-Youtube-Upload.py |
      sed -e "s:%dataType:$dataType:g" \
        -e "s:%dataArea:$dataArea:g" \
        -e "s:%dataPeriod:$dataPeriod:g" \
        -e "s:%fileInfo:$fileInfo:g" \
        -e "s:%contextPath:$contextPath:g" \
        -e "s:%title:$title:g" > ${contextPath}/Youtube-Upload.py

    #python3 ${contextPath}/Youtube-Upload.py
    /SYSTEMS/anaconda3/envs/py38/bin/python3 ${contextPath}/Youtube-Upload.py

  done
done
