load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin
;---Read data
  filename = "%fileInfo"
  fn       = addfile(filename,"r")
;  var    = fn->cors
  var    = fn->%var

var = var(::-1, :)

;var = log10(var)

; set Fill Value
; var@_FillValue = 9999
var@_FillValue = 99999999999999999999999999
var@_FillValue = 0
;var = where(var .le. 0, var, var@_FillValue)
;var = where(var .gt. 0, var, var@_FillValue)
var = where(var .gt. 0, var, var@_FillValue)

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
  wks = gsn_open_wks(wks_type, "%saveImg")

;cmap = RGBtoCmap("test_rgb.txt")  
cmap = RGBtoCmap("%cfgInfo")  
;cmap = "gsltod"
gsn_define_colormap(wks,cmap)


;---Set some resources
  res                   = True
res@mpFillOn                    = False        ; turn off gray fill

  res@mpOutlineBoundarySets       = "National"   ; turn on country boundaries
res@mpGeophysicalLineThicknessF = 5
res@mpNationalLineThicknessF = 5

  res@gsnMaximize       = True     ; maximize plot in frame
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
res@cnFillPalette        = cmap
;  res@trGridType                   = "TriangularMesh"



; res@cnLinesOn     = False
 res@cnLinesOn     = True
 res@cnInfoLabelOn = False
  res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels  = (/0, 1, 2.5, 5, 7.5, 10, 30, 50, 75, 100, 300,500,750, 1000, 3000,5000,7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000 /)

; res@cnLevelSelectionMode = "ManualLevels"
;res@cnLevelSelectionMode = "AutomaticLevels"
;  res@cnMaxLevelValF    =  1.0           ;-- maximum contour level
;  res@cnMinLevelValF    =  -1.0            ;-- maximum contour level
  res@cnMaxLevelValF    =  10000           ;-- maximum contour level
  res@cnMinLevelValF    =  0.0            ;-- maximum contour level
;  res@cnLevelSpacingF = 0.5
;res@gsnMaximize = True 
;res@cnCellFillEdgeColor = -1
;res@cnCellFillMissingValEdgeColor = -1
;res@cnFillColor = -1
res@cnConstFLabelBackgroundColor = -1

   res@gsnMaximize           =  True            ;-- maximize graphics output
   res@gsnAddCyclic          =  True            ;-- add cyclic point

  res@cnLinesOn             = False             ; turn off contour lines
  res@cnLineLabelsOn        = False             ; turn off contour labels

; ExplicitLevels
;  res@cnLevelSelectionMode  = "ExplicitLevels"
;  res@cnLevels              = levels

;  res@cnMissingValFillColor = -1             ; can also use "transparent"
; res@cnMissingValPerimOn   = True


;  res@cnLevelSelectionMode = "ExplicitLevels"
  res@gsnPaperOrientation = "auto"          
;  res@gsnMaximize         = True        ; must include w/ Paper Orientation
   
  res@cnRasterSmoothingOn = True


  plot = gsn_csm_contour_map(wks, var, res)

end
