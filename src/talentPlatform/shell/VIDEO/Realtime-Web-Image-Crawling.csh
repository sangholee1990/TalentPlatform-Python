#!/bin/csh

set dtSrtDate = `date +'%Y-%m-%d 00:00' -d '1 days ago'`
set dtEndDate = `date +'%Y-%m-%d 00:00'`

echo "[CHECK] dtSrtDate : $dtSrtDate"
echo "[CHECK] dtEndDate : $dtEndDate"

foreach rowNum(`seq 1 6`)
   #csh /SYSTEMS/PROG/SHELL/VIDEO/RunShell-Web-Image-Crawling.csh "$dtSrtDate" "$dtEndDate" UTC MP4 FALSE ${rowNum} &
   csh /SYSTEMS/PROG/SHELL/VIDEO/RunShell-Web-Image-Crawling.csh "$dtSrtDate" "$dtEndDate" KST MP4 FALSE ${rowNum} &
end

wait

bash /SYSTEMS/PROG/SHELL/VIDEO/YOUTUBE-UPLOAD/RunShell-Youtube-Upload.sh 
