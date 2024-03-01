load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin
;---Read data
  filename = "%fileInfo"
  fn       = addfile(filename,"r")
;  var    = fn->cors
;  var    = fn->%var
  var    = fn->emission

var = var(::-1, :)

; tonne to kg
;var = var(::-1, :) * 1000
;var@units = "kgCO2/day"

;var = log10(var)
;var = log(var)

; set Fill Value
var@_FillValue = 99999999999999999999999999
;var = where(var .lt. 9999, var, var@_FillValue)
;var = where(var .ge. -100000, var, var@_FillValue)
var = where(var .eq. 0, var@_FillValue, var)

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
;cmap = RGBtoCmap("%cfgInfo")  
;cmap = "gsltod"
;cmap = "BlWhRe"

;cmap = read_colormap_file("GMT_polar")
cmap = read_colormap_file("MPL_RdBu")
;cmap = read_colormap_file("BlWhRe")
;cmap = read_colormap_file("BlueWhiteOrangeRed")
cmap = cmap(::-1,:)


gsn_define_colormap(wks,cmap)

; 로그 스케일의 컬러 레벨을 생성
;levels = (/ -6000000, -3000000, -1000000, -600000, -300000, -100000, -60000, -30000, -10000, 0, 10000, 30000, 60000, 100000, 300000, 600000, 1000000, 3000000, 6000000 /)

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
res@cnLevels = (/ -30000, -19500, -6000, -3750, -750, -375, -75, -37.5, -7.5, -3.75, 0, 3.75, 7.5, 37.5, 75, 375, 750, 3750, 6000, 19500, 30000 /)
res@cnLineLabelsOn = False
;res@lbLabelStride = 21

;res@cnLineLabelsOn = True
;res@cnLabelBarEndStyle = "IncludeOuterBoxes" 

; res@cnExplicitLabelBarLabelsOn = True
res@lbLabelStrings = (/ "-30000", "-6000", "-750", "-75", "-7.5", "0", "7.5", "75", "750", "6000", "30000" /)
;res@lbLabelStrings = (/ "-30000", "", "-6000", "", "-750", "", "-75", "", "-7.5", "", "0", "", "7.5", "", "75", "", "750", "", "6000", "", "30000" /)
;res@lbLabelStrings = (/ "-30000", !, "-6000", !, "-750", !, "-75", !, "-7.5", !, "0", !, "7.5", !, "75", !, "750", !, "6000", !, "30000" /)
res@lbLabelBarOn = True

; res@lbLabelStrings = (/ "-30000", "-19500", "-6000", "-3750", "-750", "-375", "-75", "-37.5", "-7.5", "-3.75", "0", "3.75", "7.5", "37.5", "75", "375", "750", "3750", "6000", "19500", "30000" /)

;res@cnLevels = ispan(-30000, 30000, 750)
;res@cnLineLabelsOn = False

;res@cnLevels = (/ -30000, -6000, -750, -7.5, 0, 7.5, 75, 750, 6000, 30000 /)
;res@cnLineLabelsOn = True
;res@cnLineLabelsExplicit = True
;res@cnLineLabelFormat = "%i"



;res@cnLevels = ispan(-30000, 30000, 750)
;res@cnLineLabelsOn = True
;res@cnExplicitLabelBarLabelsOn = True
;res@cnExplicitLabelBarLabels = (/ -30000, -6000, -750, -7.5, 0, 7.5, 75, 750, 6000, 30000 /)


;res@cnLevels  = (/ -6000000, -5000000, -4000000, -3000000, -2000000, -1000000, -100000, -90000, -80000, -70000, -60000, -50000, -40000, -30000, -20000, -10000,-7500,-5000,-3000, -1000,-750,-500,-300, -100,-75,-50,-30, -10, -7.5,-5,-2.5, -1, 0, 1, 2.5,5,7.5, 10, 30,50,75, 100, 300,500,750, 1000, 3000,5000,7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000 /)

;res@cnLevels               = (/ -7000000, -6000000, -5000000, -4000000, -3000000, -2000000, -1000000, -900000, -800000, -700000, -600000, -500000, -400000, -300000, -200000,  -100000, -10000, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000 /)
;res@cnLevels = (/-20000000,-10000000,-30000,-20000,-10000,-7500,-5000,-3000,-1000,-750,-500,-300,-100,-75,-50,-30,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10,30,50,75,100,300,500,750,1000,3000,5000,7500,10000,20000,30000,10000000,20000000/) 
;res@cnLevels             = (/-100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000 /)
;res@cnLevels             = (/-10000, -1000, -100, -10, 0, 10, 100, 1000, 10000 /)
;res@cnLevels             = (/-1000, -100, -10, 0, 10, 100, 1000/)
;res@cnLevels             = (/-100, -10, 0, 10, 100/)
;res@cnLevels             = (/-10, 0, 10/)


;res@cnLevelSelectionMode = "AutomaticLevels"

;res@cnLevelSelectionMode = "ManualLevels"
;res@cnMaxLevelValF  =  6000000           ;-- maximum contour level
;res@cnMinLevelValF  = -6000000           ;-- maximum contour level
;res@cnMaxLevelValF  =  7000000
;res@cnMinLevelValF  = -7000000
;res@cnMaxLevelValF  =  100000
;res@cnMinLevelValF  = -100000
;res@cnMaxLevelValF  =  10000
;res@cnMinLevelValF  = -10000

;res@cnLevelSpacingF =  1000000
;res@cnLevelSpacingF =  1000000
;res@cnLevelSpacingF =    100000
;res@cnLevelSpacingF =   10000
;res@cnLevelSpacingF =   1000
;res@cnLevelSpacingF =   100

;res@lbLabelStride       = 4.0

;res@cnCellFillEdgeColor = -1
;res@cnCellFillMissingValEdgeColor = -1
;res@cnFillColor = -1
res@cnConstFLabelBackgroundColor = -1

   res@gsnMaximize           =  True            ;-- maximize graphics output
   res@gsnAddCyclic          =  True            ;-- add cyclic point

  res@cnLinesOn             = False             ; turn off contour lines
  res@cnLineLabelsOn        = False             ; turn off contour labels

;  res@cnMissingValFillColor = -1             ; can also use "transparent"
; res@cnMissingValPerimOn   = True


  res@gsnPaperOrientation = "auto"          
   
  res@cnRasterSmoothingOn = True

  plot = gsn_csm_contour_map(wks, var, res)

end
