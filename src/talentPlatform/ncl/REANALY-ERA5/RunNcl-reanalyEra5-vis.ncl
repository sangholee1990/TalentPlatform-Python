begin

    ;=============================================
    ; DOC
    ;=============================================
    ; /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/ncl/REANALY-ERA5
    ; ncl -Q RunNcl-reanalyEra5-vis.ncl

    ;=============================================
    ; 파라미터 받기
    ;=============================================
    print("[CHECK] fileInfo : " + fileInfo)
    print("[CHECK] mainTitle : " + mainTitle)
    print("[CHECK] saveImg : " + saveImg)

   ;=============================================
   ;  File Name
   ;=============================================
   ; data = asciiread("/DATA/INPUT/LSH0545/ecmwf_mean_daily_2t_out_0101.csv",(/1038240, 4/), "float")
   data = asciiread(fileInfo, (/1038240, 4/), "float")

   ;=============================================
   ;  Set Value
   ;=============================================
   nLat  = data( : , 0)
   nLon  = data( : , 1)

   ; mean value
   ; nVal  = data( : , %iColNumber) - 273.15
   nVal  = data( : , 3) - 273.15

   ;============================================
   ;  Option (res)
   ;=============================================
   wks = gsn_open_wks("png", saveImg)

   res = True
   res@gsnDraw               =  False                 ;-- don't draw the plot yet
   res@gsnFrame              =  False                 ;-- don't advance the frame
   res@txFont   = "helvetica"
   res@cnFillPalette          = "rainbow"                ;-- choose colormap
   res@gsnMaximize            =  True
   res@pmTickMarkDisplayMode  = "Always"    ; tick mark (위,경도) 표시

   ; 자동 설정
   res@cnLevelSelectionMode   = "AutomaticLevels"     ; set manual contour levels

   ; 절대 설정
   ; res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
   ; res@cnLevels    = (/0, 30, 60, 90/)   ; set levels

   ; 수동 설정
   ; res@cnLevelSelectionMode   = "ManualLevels"     ; set manual contour levels
   ; res@cnMinLevelValF         =  26           ;-- contour min. value
   ; res@cnMaxLevelValF         =  31           ;-- contour max. value
   ; res@cnMinLevelValF         =  tointeger(min(nVal))
   ; res@cnMaxLevelValF         =  tointeger(max(nVal))

   res@cnLevelSpacingF        =  0.25           ;-- contour interval(200)
   ;   res@lbLabelStride          =  3            ;-- label every 5th values on labelbar
   res@cnLinesOn = False;
   res@lbBoxLinesOn = False

   res@cnFillOn               = True               ; color fill 
   res@cnLinesOn              = False             ; no contour lines
   res@cnLineLabelsOn         = False             ; no contour labels
   res@cnFillMode             = "AreaFill"     ;-- use CellFill method
 
   res@tiMainFontHeightF      =  0.015            ;-- title font size
   res@tiMainFont             = "times-bold"
   res@gsnStringFontHeightF = 0.018              ; sub title size
   res@tmXBLabelFontHeightF = 0.017              ; xlab, ylab size
   res@lbLabelFontHeightF   = 0.017              ; colorbar font size
   res@tmXTMajorLengthF = 0                      ; tick length 
   res@trGridType = "TriangularMesh"

   ; 위경도 설정
   res@sfXArray = nLon 
   res@sfYArray = nLat 

   shpres                    =  True
   shpres@gsLineThicknessF   =  2.0                   ;-- increase line thickness
   shpres@gsLineColor        = "black"             ;-- line color

   ;============================================
   ; 맵 설정
   ;============================================
   res@mpDataBaseVersion = "MediumRes"
   res@mpOutlineOn = True;
   res@mpFillOn = False
   res@mpLandFillColor = "white"

;   res@mpMaxLatF = 35.3
;   res@mpMinLatF = 35.0

;   res@mpMaxLonF = 127.1
;   res@mpMinLonF = 126.6

   ;============================================
   ;  Title Options
   ;============================================
   res@tiMainString         =  mainTitle
   res@tiMainJust            =  "CenterLeft"
   res@tiMainPosition        =  "Left"

   ;=============================================
   ;  Plot
   ;=============================================
   plot = gsn_csm_contour_map(wks, nVal, res)

   draw(plot)
   frame(wks)

end