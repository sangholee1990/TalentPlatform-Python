load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin
;---Read data
  filename = "%fileInfo"
  fn       = addfile(filename, "r")
;  var    = fn->cors
;  tmp    = fn->%var
  var    = fn->%var

var = var(::-1, :)
;var = var
printVarSummary(var)

;dims = dimsizes(tmp)
;var = onedtond(ndtooned(tmp), (/dims(1), dims(2)/) )
printVarSummary(var)

; set Fill Value
var@_FillValue = 10000000
var = where(var .lt. 10000000, var, var@_FillValue)
var = where(var .eq. 0, var@_FillValue, var)


var = log10(var)

;---Open workstation and change color map
;   wks_type = "ps"
   wks_type = "png"
  wks_type@wkOrientation = "landscape"
;  wks_type@wkPaperWidthF  = 12  ; in inches
;  wks_type@wkPaperHeightF = 8  ; in inches
;  wks_type@wkWidth  = 7292  ; in inches
;  wks_type@wkHeight = 7292  ; in inches
  wks_type@wkWidth  = 10000  ; in inches
  wks_type@wkHeight = 10000  ; in inches
;  wks_type@wkWidth  = 1000  ; in inches
;  wks_type@wkHeight = 1000  ; in inches
  wks = gsn_open_wks(wks_type, "%saveImg")

;cmap = RGBtoCmap("test_rgb.txt")  
;cmap = RGBtoCmap("%cfgInfo")  
;cmap = "gsltod"
cmap = "MPL_RdBu"
;cmap = "MPL_coolwarm"
gsn_define_colormap(wks, cmap)
gsn_reverse_colormap(wks)
;gsn_draw_colormap(wks)


;---Set some resources
  res                   = True
res@mpFillOn                    = False        ; turn off gray fill

  res@gsnDraw              = False
  res@gsnFrame             = False

  res@mpOutlineBoundarySets       = "National"   ; turn on country boundaries
res@mpGeophysicalLineThicknessF = 5
res@mpNationalLineThicknessF = 5

  res@gsnMaximize       = True     ; maximize plot in frame
;  res@gsnMaximize       = False     ; maximize plot in frame
;  res@tiMainString      = filename
;  res@gsnLeftString     = "Aerosol Optical Thickness"
;  res@gsnRightString    = "0.55 microns"

;res@cnFillPalette        = colors
  res@cnFillOn          = True            ; turn on contour fill
; rest@cnLinesOn                   = False            ; no contour lines
; rest@cnLineLabelsOn                   = False            ; no contour lines
;  res@cnFillMode        = "RasterFill"    ; faster than default "AreaFill"
  res@cnFillMode        = "CellFill"    ; faster than default "AreaFill"
;  res@cnFillMode        = "MeshFill"    ; faster than default "AreaFill"
;res@cnFillPalette        = cmap
;  res@trGridType                   = "TriangularMesh"



; res@cnLinesOn     = False
 res@cnLinesOn     = True
 res@cnInfoLabelOn = False
;  res@cnLevelSelectionMode = "ExplicitLevels"
;  res@cnLevelSelectionMode = "AutomaticLevels"
;  res@cnLevelSelectionMode = "ManualLevels"
;  res@cnMaxLevelValF    =  10000           ;-- maximum contour level
;  res@cnMinLevelValF    =  0            ;-- maximum contour level
;  res@lbLabelStride = 1
;  res@cnLevelSpacingF = 1000
;res@cnCellFillEdgeColor = -1
;res@cnCellFillMissingValEdgeColor = -1
;res@cnFillColor = -1
res@cnConstFLabelBackgroundColor = -1

 res@cnLinesOn     = True
 res@cnInfoLabelOn = False
  res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels  = (/0, 1, 2.5, 5, 7.5, 10, 30, 50, 75, 100, 300,500,750, 1000, 3000,5000,7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000 /)
res@cnLabelBarEndStyle = "IncludeOuterBoxes"



   res@gsnAddCyclic          =  True            ;-- add cyclic point
  res@cnLinesOn             = False             ; turn off contour lines
  res@cnLineLabelsOn        = False             ; turn off contour labels
;  res@cnLevels              = levels
;  res@cnMissingValFillColor = -1             ; can also use "transparent"
; res@cnMissingValPerimOn   = True
  res@gsnPaperOrientation = "auto"          
  res@cnRasterSmoothingOn = True

  plot = gsn_csm_contour_map(wks, var, res)


;-- polyline
  lon = (/ 6.,  50.,  50.,   6.,   6./)
  lat = (/47.5, 47.5, 80, 80, 47.5/)
  
;-- polyline resources
  pres                       =  True
;  pres@gsFillColor      = "Transparent"
  pres@gsLineThicknessF       =  5       ;-- set line thickness
  pres@gsLineColor            = "black"       ;-- set line color

  ;dum = gsn_add_polyline(wks, plot, (/-45, -45,  5,   5, -45/), (/-25,  35, 35, -25, -25/), pres)

; 아시아
minLon = 64.35419
maxLon = 145.21390
minLat = 4.38144
maxLat = 55.02741
dum5 = gsn_add_polyline(wks, plot, (/minLon, minLon, maxLon, maxLon, minLon/), (/minLat,  maxLat, maxLat, minLat, minLat/), pres)

; 유럽
minLon = -14.00478
maxLon = 44.71888
minLat = 34.430346
maxLat = 60.090
dum = gsn_add_polyline(wks, plot, (/minLon, minLon, maxLon, maxLon, minLon/), (/minLat,  maxLat, maxLat, minLat, minLat/), pres)

; 아프리카
minLon = -23.08840
maxLon = 51.44285
minLat = -34.41006
maxLat = 37.61990
dum2 = gsn_add_polyline(wks, plot, (/minLon, minLon, maxLon, maxLon, minLon/), (/minLat,  maxLat, maxLat, minLat, minLat/), pres)

; 남아메리카
minLon = -85.35378
maxLon = -33.94776
minLat = -56.24722
maxLat = 14.72564
dum3 = gsn_add_polyline(wks, plot, (/minLon, minLon, maxLon, maxLon, minLon/), (/minLat,  maxLat, maxLat, minLat, minLat/), pres)

; 미국
minLon = -124.74968
maxLon = -63.57780
minLat = 24.55069
maxLat = 50.05260
dum4 = gsn_add_polyline(wks, plot, (/minLon, minLon, maxLon, maxLon, minLon/), (/minLat,  maxLat, maxLat, minLat, minLat/), pres)

  draw(plot)
  frame(wks)

end
