#!/bin/csh

# csh Realtime-Schedule-Web-Image-Crawling.csh 2020-01-09 00:00 2020-01-09 01:00 UTC MP4 FALSE 1
# nohup csh Realtime-Schedule-Web-Image-Crawling.csh 2020-01-09 15:00 2020-01-09 16:00 KST MP4 FALSE 1 &

# TEST
# nohup csh Realtime-Schedule-Web-Image-Crawling.csh 2021-01-30 15:00 2021-01-31 15:00 UTC MP4 FALSE 1 &
# nohup csh Realtime-Schedule-Web-Image-Crawling.csh 2021-01-31 15:00 2021-02-01 15:00 KST MP4 FALSE 1 &
# nohup csh Realtime-Schedule-Web-Image-Crawling.csh 2021-02-01 15:00 2021-02-02 15:00 KST MP4 FALSE 1 &
# nohup csh Realtime-Schedule-Web-Image-Crawling.csh 2021-02-02 15:00 2021-02-03 15:00 KST MP4 FALSE 1 &

set sPid = `echo $$`
set sPresentTime = `date +"%Y%m%d%H%M"`
set dtYmd = `date +"%Y%m/%d"`

echo "pid : " $sPid
echo "presentTime : " $sPresentTime

# set sContextPath = `pwd`
# set sContextPath = `echo /SYSTEMS/PROG/ShellScript/VIDEO`
#set sContextPath = `echo /SYSTEMS/PROG/ShellScript/VIDEO`
set sContextPath = `echo /SYSTEMS/PROG/SHELL/VIDEO`
set sTmpContextPath = `echo ${sContextPath}/TMP/${sPid}`
set sLogContextPath = `echo ${sContextPath}/LOG`
set sInputContextPath = `echo ${sContextPath}/INPUT`
# set sOutputContextPath = `echo ${sContextPath}/OUTPUT`
# set sOutputContextPath = `echo /home/fedora/OUTPUT/VIDEO/${dtYmd}`
#set sOutputContextPath = `echo /SYSTEMS/OUTPUT/VIDEO/${dtYmd}`
set sOutputContextPath = `echo /DATA/OUTPUT/VIDEO/${dtYmd}`

if (! -d $sTmpContextPath) then
   mkdir -p $sTmpContextPath
else
   rm -rf ${sTmpContextPath}/*
endif

if (! -d $sLogContextPath) then
   mkdir -p $sLogContextPath
endif

if (! -d $sInputContextPath) then
   mkdir -p $sInputContextPath
endif

if (! -d $sOutputContextPath) then
   mkdir -p $sOutputContextPath
endif

set sInputDirName         = `echo ${sInputContextPath}/Image_Url_For_INPUT.inp`
set sInputMetaDataDb      = `echo ${sInputContextPath}/Meta_Data.db`
set sExecutionLogDirName  = `echo ${sLogContextPath}/Execution_${sPresentTime}.log`
set sErrorLogDirName      = `echo ${sLogContextPath}/Error_${sPresentTime}.log`
   
echo [`date "+%Y/%m/%d %T"`] "=================================================  [START]  Realtime_Schedule_Web_Image_Crawling.csh  ==================================================" > $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "=================================================  [START]  Realtime_Schedule_Web_Image_Crawling.csh  ==================================================" > $sErrorLogDirName

cat $sInputMetaDataDb | egrep -v '(^[[:space:]]*$|#)' > $sInputDirName

set sArgv = `echo $1`
set sArgv2 = `echo $2`
set sArgv3 = `echo $3`
set sArgv4 = `echo $4`
set sArgv5 = `echo $5`
set sArgv6 = `echo $6`
# set sArgv7 = `echo $7`
# set sArgv8 = `echo $8`

echo [`date "+%Y/%m/%d %T"`] "sArgv : $sArgv" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "sArgv2 : $sArgv2" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "sArgv3 : $sArgv3" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "sArgv4 : $sArgv4" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "sArgv5 : $sArgv5" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "sArgv6 : $sArgv6" >> $sExecutionLogDirName
# echo [`date "+%Y/%m/%d %T"`] "sArgv7 : $sArgv7" >> $sExecutionLogDirName
# echo [`date "+%Y/%m/%d %T"`] "sArgv8 : $sArgv8" >> $sExecutionLogDirName

# echo "sArgv : $sArgv"
# echo "sArgv2 : $sArgv2"
# echo "sArgv3 : $sArgv3"
# echo "sArgv4 : $sArgv4"
# echo "sArgv5 : $sArgv5"
# echo "sArgv6 : $sArgv6"

if (${#sArgv} == 0) then
   set sStartTime = `date +"%Y-%m-%d 15:00"`
else
   set sStartTime   = `date -d "${sArgv}" +"%Y-%m-%d %H:%M"`
endif

if (${#sArgv2} == 0) then
   set sEndTime = `date +"%Y-%m-%d 15:00" -d "1 day"`
else
   set sEndTime   = `date -d "${sArgv2}" +"%Y-%m-%d %H:%M"`
endif

if (${#sArgv3} == 0 || $sArgv3 == "UTC") then
   set sConv = "UTC"
else 
   set sConv = "KST"
endif

if (${#sArgv4} == 0 || $sArgv4 == "MP4") then
   set sIsMp4 = "true"
else 
   set sIsMp4 = "false"
endif

if (${#sArgv5} == 0 || $sArgv5 == "GIF") then
   set sIsGif = "true"
else 
   set sIsGif = "false"
endif

if (${#sArgv6} == 0) then
   set selRow = "1"
else
   set selRow = `echo ${sArgv6}`
endif

echo [`date "+%Y/%m/%d %T"`] "[$sStartTime - $sEndTime $sConv] VIDEO : $sIsMp4 | GIF : $sIsGif | ROW : $selRow" >> $sExecutionLogDirName

foreach sInputRow("`cat $sInputDirName | sed -n ${selRow}p`")

   set sStartTimeUnix = `date -d "${sStartTime} ${sConv}" +"%s"`
   set sEndTimeUnix   = `date -d "${sEndTime} ${sConv}" +"%s"`
   set sIncreTimeUnix = $sStartTimeUnix
   
   set sStartDateTime = `date -d "${sStartTime} ${sConv}" +"%Y%m%d%H%M"`
   set sEndDateTime = `date -d "${sEndTime} ${sConv}" +"%Y%m%d%H%M"`
   
   set sInputRowUrlDir  = `echo $sInputRow | awk '{print $1}'`
   set sInputRowUrlName = `echo $sInputRow | awk '{print $2}'`
   set sInterval        = `echo $sInputRow | awk '{print $3}'`
   set sIntervalUnit    = `echo $sInputRow | awk '{print $4}'`
  
   set sOutputGifDirName = `echo ${sOutputContextPath}/Gif_For_OUTPUT_${sInputRowUrlName}_${sPresentTime}_${sStartDateTime}-${sEndDateTime}_${sConv}.gif`
   set sOutputMp4DirName = `echo ${sOutputContextPath}/Mp4_For_OUTPUT_${sInputRowUrlName}_${sPresentTime}_${sStartDateTime}-${sEndDateTime}_${sConv}.mp4`

   while ($sIncreTimeUnix <= $sEndTimeUnix)
   	  set sYear   = `date +"%Y" -d @$sIncreTimeUnix`
      set sMonth  = `date +"%m" -d @$sIncreTimeUnix`
      set sDay    = `date +"%d" -d @$sIncreTimeUnix`
   	  set sHour   = `date +"%H" -d @$sIncreTimeUnix`
   	  set sMinute = `date +"%M" -d @$sIncreTimeUnix`
   
      set sUrlDir  = `echo ${sInputRowUrlDir} | sed -e "s:%Y:${sYear}:g" -e "s:%m:${sMonth}:g" -e "s:%d:${sDay}:g" -e "s:%H:${sHour}:g" -e "s:%M:${sMinute}:g"`
   	  set sUrlName = `echo ${sInputRowUrlName} | sed -e "s:%Y:${sYear}:g" -e "s:%m:${sMonth}:g" -e "s:%d:${sDay}:g" -e "s:%H:${sHour}:g" -e "s:%M:${sMinute}:g"`
      set sTmpUrlDirName = `echo ${sUrlDir}/${sUrlName}`
      set sExt = `echo $sTmpUrlDirName:e`
   	  set sTmpDirName = `echo ${sTmpContextPath}/${sUrlName}`
      set isImageDown = "F"
      set isImageSize = "F"

      @ sIncreTimeUnix = `date -d "${sYear}${sMonth}${sDay} ${sHour}${sMinute} ${sInterval} ${sIntervalUnit}" +%s`

      if (! -e $sTmpDirName) then

	#/usr/local/bin/wget -t 100 -cnv -P $sTmpContextPath $sTmpUrlDirName
	wget -t 100 -cnv -P $sTmpContextPath $sTmpUrlDirName

	if (! -e $sTmpContextPath/$sTmpUrlDirName:t) then
           continue
	endif

         set isImageDown = "S"
   	 set sImageSize = `stat -c "%s" $sTmpDirName`

         if ($sImageSize > 1195) then
            set isImageSize = "S"
         else
            echo [`date "+%Y/%m/%d %T"`] $sArgv $sArgv2 $sArgv3 $sArgv4 $sArgv5 $sArgv6 $sYear $sMonth $sDay $sHour $sOutputMp4DirName >> $sErrorLogDirName
            rm -f $sTmpDirName
         endif
        else
         echo [`date "+%Y/%m/%d %T"`] $sArgv $sArgv2 $sArgv3 $sArgv4 $sArgv5 $sArgv6 $sYear $sMonth $sDay $sHour $sOutputMp4DirName >> $sErrorLogDirName
    	endif
      
      echo [`date "+%Y/%m/%d %T"`] $sTmpUrlDirName $sTmpDirName $sImageSize $isImageDown $isImageSize >> $sExecutionLogDirName
   end

   if ($sIsMp4 == "true") then
      if (! -e $sOutputMp4DirName) then
#         ffmpeg -y -framerate 10 -f image2 -pattern_type glob -i "${sTmpContextPath}/*.${sExt}" -vcodec libx264 -pix_fmt yuv420p -r 30 -vf "scale =trunc(iw/2)*2:trunc(ih/2)*2" $sOutputMp4DirName
         ffmpeg -y -framerate 7.5 -f image2 -pattern_type glob -i "${sTmpContextPath}/*.${sExt}" -vcodec libx264 -pix_fmt yuv420p -r 30 -vf "scale =trunc(iw/2)*2:trunc(ih/2)*2" $sOutputMp4DirName
      
         echo `[date "+%Y/%m/%d %T"]` $sOutputMp4DirName "S" >> $sExecutionLogDirName
      else
         echo `[date "+%Y/%m/%d %T"]` $sOutputMp4DirName "F" >> $sExecutionLogDirName
         echo `[date "+%Y/%m/%d %T"]` $sArgv $sArgv2 $sArgv3 $sArgv4 $sArgv5 $sArgv6 $sYear $sMonth $sDay $sHour $sOutputMp4DirName >> $sErrorLogDirName
      endif
   endif
   
   if ($sIsGif == "true") then
      if (! -e $sOutputGifDirName) then
      	 # convert -loop 0 -delay 10 `ls ${sTmpContextPath}/*` $sOutputGifDirName
      	
         # 0.5 sec
      	 convert -loop 0 -delay 50 `ls ${sTmpContextPath}/*.${sExt}` $sOutputGifDirName
      	
         echo [`date "+%Y/%m/%d %T"`] $sOutputGifDirName "S" >> $sExecutionLogDirName
      else 
        	echo [`date "+%Y/%m/%d %T"`] $sOutputGifDirName "F" >> $sExecutionLogDirName
        	echo [`date "+%Y/%m/%d %T"`] $sArgv $sArgv2 $sArgv3 $sArgv4 $sArgv5 $sArgv6 $sYear $sMonth $sDay $sHour $sOutputGifDirName >> $sErrorLogDirName
      endif
   endif

   rm -rf ${sTmpContextPath}/*

end

echo [`date "+%Y/%m/%d %T"`] "=================================================  [END]  Realtime_Schedule_Web_Image_Crawling.csh  ==================================================" >> $sExecutionLogDirName
echo [`date "+%Y/%m/%d %T"`] "=================================================  [END]  Realtime_Schedule_Web_Image_Crawling.csh  ==================================================" >> $sErrorLogDirName
