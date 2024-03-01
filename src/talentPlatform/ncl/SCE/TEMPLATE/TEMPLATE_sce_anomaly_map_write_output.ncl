;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; extract sce value at given point
;;; plot monthly anomaly map
;;; data: self-made monthly snow cover extent from Rugster University
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shapefile_utils.ncl"

begin

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define analysis period
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 현재 연도 및 월 설정
; preYear = 2014
; preMonth = 4
preYear = %preYear
preMonth = %preMonth

; 아노말리 시작/종료 연도 및 월 설정
srtYear = 1967 ;1981
endYear = 2021 ;2010

srtMonth = 04
endMonth = 04

; extract value at point
pLat = 60
pLon = 90

; domain
LatMin = 55
LatMax = 65
LonMin = 85
LonMax = 95

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; set environment variables
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; inFile   = "./SCE_monthly_nhsce_v01r01_19661004_20211004.nc"
inFile   = "%inFile"
maskFile = "$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc"
; saveFile = "SCE_Point_Data_60N_90E_"+srtYear+"_"+endYear+"_"+srtMonth+".csv"
saveFile = "%saveFile"

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
; option 0
;(89,358)	0.07986836
;(89,359)	0.07896777

; option 1
;(29,358)	0.09070244
;(29,359)	0.09644037

; option 2
;(29,88)	0.1539322
;(29,89)	0.1681514


;-- extract current time index for the present period
timeIdxPre = ind(\
	  (year .eq. preYear) \
	  .and. (month .eq. preMonth) \
	  )
print(timeIdxPre)	; 567th

;--- calculate lat/lon average of present data
;preVal = dim_avg_n_Wrap(val(timeIdxPre,:,:), (/0/))	; 1d array
preVal2d = val(timeIdxPre,:,:)				; 2d array

;--- calculate anomaly
anoVal = (preVal2d - climVal)*100
;print(anoVal)
;exit

;--- masking land only
anoVal = mask(anoVal, lsmData.eq.0, False)

;--- give coordinate information for map plot
anoVal!0 = "lat"
anoVal!1 = "lon"
anoVal&lat = lat
anoVal&lon = lon
anoVal&lat@units = "degrees_north"
anoVal&lon@units = "degrees_east"

;printVarSummary(anoVal)
;Coordinates:
;            lat: [ 0..89]
;            lon: [-180..179]
;Number Of Attributes: 1
;  _FillValue :	9.96921e+36
;printMinMax(anoVal, 0)	; min=-82.967   max=64.1349	--> snow cover % 아노말리 값의 범위


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; write mean sce data over 55-65N & 85-95E as csv
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
preVal2d_siberia = dim_avg_n_Wrap(val(timeIdxPre,{55:65},{85:95}), (/0/))				; 2d array
meanVal = dim_avg(preVal2d_siberia)

; preVal2d_siberia = dim_avg(val(timeIdxPre,{55:65},{85:95}))

; print(preVal2d_siberia)
; print(timeIdxPre)
; printVarSummary(preVal2d_siberia)
; printVarSummary(timeIdxPre)
; printVarSummary(timeIdx)
; printVarSummary(meanVal)

; write_table(saveFile, "w", \
; 	[/ year(timeIdxPre), month(timeIdxPre), preVal2d_siberia /], \
; 	 "%i, %i, %f" \
;	)

write_table(saveFile, "w", \
	[/ year(timeIdxPre), month(timeIdxPre), meanVal /], \
	 "%i, %i, %f" \
	)

end
