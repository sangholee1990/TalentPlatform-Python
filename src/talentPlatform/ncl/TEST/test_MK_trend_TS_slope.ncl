begin

; 그리고 ncl 작업의뢰할 내용은 과거에 한번 의뢰했던 내용인데 
; 약간 수정하고 싶어서요. Mann-Kendal 검정으로 격자별 추세가 있는지 여부를 
; 테스트한 후 Teil-Sen방법으로 격자마다 기울기를 구하는데 
; p값이 0.05 이하인 유의한 지역만 점으로 나타내는 것입니다. 
; p값은 ttest로 하면 될것 같아요(https://www.ncl.ucar.edu/Document/Functions/Built-in/ttest.shtml). 
; 현 스크립트는 trend_manken에서 옵션을 0,1로만 바꾸는 것만 가능한데 
; 그림이 같이 나오지 않습니다.


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; set environment variables
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
filepath = "./"

; 참조 마스킹 P값
refMaskPval = 0.05

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; read NetCDF file
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
dat_t=addfile(filepath+"Annual_total_DJFsd_MAMsm_MAMt2m_2001_2021.nc","r")	; lon 192 x lat 94
print(dat_t)

; extract time
ori_time = dat_t->time
ori_time@units = "days since 2000-01-01 00:00:00"
time 	 = ut_calendar(ori_time, 0)		; float type
year 	 = tointeger(time(:,0))
month 	 = tointeger(time(:,1))
day	 = toint(time(:,2))
ddd	 = day_of_year(year, month, day)
;print(year+" " + month+" "+day+" "+ddd)

; extract variable
val   = dat_t->MAMt2m
val@_FillValue = -9.e+33
;printMinMax(val,True)

; 2023.03.22
;lat   = dat_t->latitude
;lon   = dat_t->longitude
lat   = dat_t->lat
lon   = dat_t->lon

;--- trend of MAMt2m, 2001-2021


; 2023.03.22 참조 
; https://www.ncl.ucar.edu/Document/Functions/Built-in/trend_manken.shtml

; opt = False
opt = True
rc  = trend_manken(val, opt, 0)
printVarSummary(rc)

;printVarSummary(rc)
;   rc(0) = significance ; (i.e., 1-p_value (two-sided))
;   rc(1) = trend

copy_VarCoords(val(0,:,:),rc(0,:,:))
copy_VarCoords(val(0,:,:),rc(1,:,:))

; 격자별 추세
; 2023.03.22 반환 결과 (1) : Theil-Sen 선형 추세 추정치(Sen 추세 추정치)
trend = rc(1,:,:)	;기울기가 거의 없다고 보면됨

; 2023.03.22 반환 결과 (0) : Mann-Kendall 추세 유의성
pvalue = rc(0,:,:)
;printMinMax(trend,0)	;min=-0.107356   max=0.301073
;printMinMax(pvalue,0)	;min=0   max=0.999998

; 2023.03.22 마스킹 처리
; Teil-Sen방법으로 격자마다 기울기를 구하는데 p값이 0.05 이하인 유의한 지역만 점으로 나타내는 것입니다. 
maskTrend = trend
maskTrend = mask(maskTrend, pvalue .gt. refMaskPval, False)
printMinMax(maskTrend, 0)

maskPvalue = pvalue
maskPvalue = mask(maskPvalue, pvalue .gt. refMaskPval, False)
printMinMax(maskPvalue, 0)

; p값은 ttest로 하면 될것 같아요(https://www.ncl.ucar.edu/Document/Functions/Built-in/ttest.shtml).
; 두 샘플 간의 평균 차이 검정 방법?

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; plotting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
figdir  = "./"
figname = figdir+"2.ERA5_MAMt2m_sen_slope_trend_map_2001_2021"

wks = gsn_open_wks("eps", figname)       ; export png file

cmap = read_colormap_file("hotcold_18lev")
gsn_define_colormap(wks,"hotcold_18lev")

subTitle = "[ MAM, 2001-2021 ]"
cbTitle = "Temperature trend [ERA5]"


res = True

res@gsnMaximize         = True
res@gsnFrame            = False
res@gsnDraw             = False
res@gsnAddCyclic        = False

res@cnFillOn            = True          ; turn on color
res@cnLinesOn           = False         ; no cn lines
res@cnFillMode          = "RasterFill"  ; assigning colors to the elements of a 2D array of rectangular cel
res@cnRasterSmoothingOn = True          ; take less time to plot, version 6.6.2, raster is not smoothing

res@cnFillOn             = True
res@cnLinesOn            = False
res@cnLineLabelsOn       = False
;res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
;res@cnMinLevelValF       =  -0.2                ; set min contour level
;res@cnMaxLevelValF       =  0.2                 ; set max contour level
;res@cnLevelSpacingF      =  0.05               ; set contour spacing

; 2023.03.22 제외  
; res@cnLevelSelectionMode = "ExplicitLevels"     ; set manual contour levels
; res@cnLevels    = (/ -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2/)   ; set levels
; res@cnFillColors = (/2,   3,     4,   7,      11, 11, 14,   15,   18, 19/)  ; set the colors to

;p-value
;res@cnLevels    = (/ 0.00001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1./)   ; set levels
;res@cnFillColors = (/2,     80,    90,   95,    97,    10,  4,   2, 2/)  ; set the colors to

res@lbLabelAutoStride    = True               ; optimal label stride
res@lbLabelFont          = 25.
res@lbLabelFontHeightF   = 0.018
res@lbTitleOn            = True
res@lbTitleString        = "[oC/year]"
;res@lbTitleString        = "[p-value]"	;"[oC/year]"  ;"[p-value]"
res@lbTitlePosition      = "bottom"
res@lbTitleOffsetF       = 0.025
res@lbTitleExtentF       = 0.15
res@lbTitleFont          = 25.
res@lbTitleFontHeightF   = 0.015
res@lbOrientation        = "horizontal"
res@lbBoxLinesOn         = False
res@lbLabelStride        = 2

 res@pmLabelBarWidthF     = 0.8
 res@pmLabelBarHeightF    = 0.08
 res@pmLabelBarOrthogonalPosF = 0.3
 res@pmLabelBarParallelPosF = 0.5

 minlat		    = 50.
 maxlat             = 80.                ; max lat to mask
 minlon             = 20.                 ; min lon to mask
 maxlon             = 180.                ; max lon to mask

 res@mpMinLatF             = minlat
 res@mpMaxLatF             = maxlat
 res@mpMinLonF             = minlon
 res@mpMaxLonF             = maxlon

res@mpFillOn              = False
res@pmTickMarkDisplayMode = "Always"

res@gsnRightString              = subTitle
res@gsnRightStringFontHeightF   = 0.015
res@gsnLeftString               = cbTitle
res@gsnLeftStringFontHeightF    = 0.015

res@mpLandFillColor       = False
res@mpDataBaseVersion     = "Ncarg4_1"
res@mpDataSetName         = "Earth..4"
res@mpDataResolution      = "MediumResolution"
res@mpOutlineBoundarySets = "National"

; label style
res@tmYLLabelFont        = 25.
res@tmYRLabelFont        = 25.
res@tmXBLabelFont        = 25.
res@tmXBLabelFontHeightF = 0.02
res@tmYLLabelFontHeightF = 0.02

  ; title font style on x-axis and y-axis
  res@tiYAxisFont = 25.
  res@tiXAxisFont = 25.

; plot = gsn_csm_contour_map(wks, trend, res)
plot = gsn_csm_contour_map(wks, maskTrend, res)

;plot = gsn_csm_contour_map(wks, pvalue, res)
; plot = gsn_csm_contour_map(wks, maskPvalue, res)


  resP                      = True            ; modify the panel plot
  resP@gsnMaximize          = True            ; large format
  resP@gsnPaperOrientation  = "portlait"
  resP@txString             = ""

gsn_panel(wks, plot, (/1,1/), resP)

delete(wks)
system("convert -density 150x150 -trim -rotate -90 "+figname+".eps "+figname+".png")
system("rm -rf "+figname+".eps")

end