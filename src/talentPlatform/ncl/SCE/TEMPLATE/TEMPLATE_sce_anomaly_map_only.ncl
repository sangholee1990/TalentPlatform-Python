;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; extract sce value at given point
;;; plot monthly anomaly map
;;; data: self-made monthly snow cover extent from Rugster University
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define analysis period
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 현재 연도 및 월 설정
preYear = 2014
preSrtMonth = 04
preEndMonth = 05
;preYear = %preYear
;preSrtMonth = %preSrtMonth
;preEndMonth = %preEndMonth

; 아노말리 시작/종료 연도 및 월 설정
srtYear = 2000 ;1981
endYear = 2020 ;2010

srtMonth = 04
endMonth = 05

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; set environment variables
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; inFile   = "./SCE_monthly_nhsce_v01r01_19661004_20211004.nc"
inFile   = "%inFile"
maskFile = "$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc"
saveFile = "SCE_Point_Data_60N_90E_"+srtYear+"_"+endYear+"_"+srtMonth+".csv"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; read NetCDF file 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; snow cover extent data
snow = addfile(inFile, "r")

; extract time
ori_time = snow->time
time 	 = ut_calendar(ori_time, 0)		; float type
year 	 = tointeger(time(:,0))
month 	 = tointeger(time(:,1))
day	 = toint(time(:,2))
ddd	 = day_of_year(year, month, day)
;print(year+" " + month+" "+day+" "+ddd)


; extract variable
val = snow->snow_grd
lat = snow->latitude
lon = snow->longitude


; read mask file
lsId    = addfile(maskFile, "r")
lsmData = landsea_mask(lsId->LSMASK, lat, lon)
;printVarSummary(lsmData)
;printMinMax(lsmData, 0)	;min=0   max=3 --> original data has 4=Ice Shelf but it my snow cover data covers only NH



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; anomaly data with polar projection 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;-- extract climatology time index for 04,1981-2010 climatology
timeIdx = ind(\ 
	(year .ge. srtYear) .and. (year .le. endYear) \
	.and. (month .ge. srtMonth) .and. (month .le. endMonth) \
	)
;print(timeIdx)	; ith
;printVarSummary(timeIdx)
;printMinMax(timeIdx, 0)

;--- calculate snow cover extent climatology for April, 1981-2010
climVal = dim_avg_n_Wrap(val(timeIdx,:,:),(/0/))
;printVarSummary(climVal)
;print(climVal)
;exit

;-- extract current time index for the present period
;timeIdxPre = ind(\
;	  (year .eq. preYear) \
;	  .and. (month .ge. preSrtMonth) \
;	  .and. (month .le. preEndMonth) \
;	  )
timeIdxPre = ind( %timeIdxPre )
; print(timeIdxPre)	; 567th
;print(year(timeIdxPre))
;print(month(timeIdxPre))
; printVarSummary(timeIdxPre)

;--- calculate lat/lon average of present data
preVal = dim_avg_n_Wrap(val(timeIdxPre,:,:), (/0/))	; 1d array
;preVal2d = val(timeIdxPre,:,:)				; 2d array
;printVarSummary(preVal2d)
; printVarSummary(preVal)

;--- calculate anomaly
;anoVal = (preVal2d - climVal)*100				
anoVal = (preVal - climVal)*100				

;--- masking land only
anoVal = mask(anoVal, lsmData.eq.0, False)

;--- give coordinate information for map plot
anoVal!0 = "lat"
anoVal!1 = "lon"
anoVal&lat = lat
anoVal&lon = lon
anoVal&lat@units = "degrees_north"
anoVal&lon@units = "degrees_east"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; plotting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; figname = "test_sce_anomaly_"+preYear+preSrtMonth+preEndMonth
figname = "%figname"

wks = gsn_open_wks("eps", figname)       ; export png file
cmap = read_colormap_file("temp_diff_18lev")
gsn_define_colormap(wks,"temp_diff_18lev")

; subTitle = "[ "+ preYear + "," + preSrtMonth + "-" + preEndMonth +"]"
subTitle = "[ "+ %dtSrtYm + "-" + %dtEndYm +"]"
cbTitle = "SCE anomaly (%)"

res = True

res@gsnMaximize         = True
res@gsnPolar            = "NH"
res@gsnFrame            = False
res@gsnDraw             = False

res@gsnRightString      = subTitle
res@cnFillOn            = True          ; turn on color
res@cnLinesOn           = False         ; no cn lines
res@cnFillMode          = "RasterFill"  ; assigning colors to the elements of a 2D array of rectangular cel
res@cnRasterSmoothingOn = True          ; take less time to plot, version 6.6.2, raster is not smoothing

res@cnFillOn             = True
res@cnLinesOn            = False
res@cnLineLabelsOn       = False
res@cnLevelSelectionMode = "ExplicitLevels"     ; set manual contour levels
res@cnLevels             = (/-5, -4, -3, -2, -1, -0.5, 0.5,  1,  2,   3,  4, 5/) 
res@cnFillColors         = (/   3,    4, 5,  7,    8,  10,   0,   12, 14,   15, 16, 17, 18/)

res@mpCenterLonF          = 90.               ; center plot at 180
res@mpMinLatF             = 40. 
res@mpFillOn              = False

res@gsnRightString              = subTitle
res@gsnRightStringFontHeightF   = 0.015
res@gsnLeftString               = "[Climatology: "+srtYear+"-"+endYear+"]"
res@gsnLeftStringFontHeightF    = 0.015

res@mpLandFillColor       = False
res@mpDataBaseVersion     = "Ncarg4_1"
res@mpDataSetName         = "Earth..4"
res@mpDataResolution      = "MediumResolution"
res@mpOutlineBoundarySets = "National"

res@lbOrientation         = "vertical"
res@lbBoxLinesOn          = False

plot = gsn_csm_contour_map(wks, anoVal, res)

; mark my station
  mkres = True
  mkres@gsMarkerIndex = 3
  mkres@gsMarkerSizeF = 0.015
  mkres@gsMarkerColor = "black"
  mkres@gsMarkerThicknessF = 5

  dum = gsn_add_polymarker(wks, plot, 89.21, 60.8, mkres)

  resP                      = True            ; modify the panel plot
  resP@gsnMaximize          = True            ; large format
  resP@gsnPaperOrientation  = "portlait"
  resP@txString             = ""
  resP@pmLabelBarWidthF     = 0.7
  resP@gsnPanelXWhiteSpacePercent = 7
  resP@gsnPanelRowSpec      = True

gsn_panel(wks, plot, (/1,1/), resP)

delete(wks)
system("convert -density 150x150 -trim -rotate -90 "+figname+".eps "+figname+".png")
system("rm -rf "+figname+".eps")

end
