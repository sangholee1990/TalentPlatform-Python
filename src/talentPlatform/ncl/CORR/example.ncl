;================================================
;  topo_2.ncl
;================================================
; Concepts illustrated:
;   - Drawing a topographic map using 5' data
;   - Drawing topographic data using a custom color map
;   - Adding labels to both ends of a labelbar
;   - Using "MeshFill" for faster contouring
;   - Reading binary data using "cbinread"
;   - Changing the byte order when reading binary data
;   - Using functions for cleaner code
;----------------------------------------------------------------------
; This script draws the full 5' (now deprecated according to the
; website) topo grid downloaded from:
;
;  http://www.ngdc.noaa.gov/mgg/global/relief/ETOPO5/TOPO/ETOPO5/
;
; Other topo files can be found at: http://www.ngdc.noaa.gov/mgg/topo/
;
; This TOPO file is a binary file. See below for details.
;----------------------------------------------------------------------
; The data file is formatted as 16-bit BINARY INTEGERS in two byte 
; order; the file ETOPO5.DAT is in "normal," or hi-byte-first 
; order, as used by Macintosh, Sun, and some other workstations. 
; There are 2160x4320 data values, one for each five minutes of latitude
; and longitude, for a total of 9,331,200 points or 18,662,400 bytes. 
; Data values are in whole meters, representing the elevation of the 
; CENTER of each cell.
; 
; Data Order in the Files:
; 
; The file may be thought of as having a logical record size of 
; 8640 bytes.  The data start at the North Pole (90 deg N, 0 deg 0' 
; E) and are arranged in bands of 360 degrees x 12 points/degree = 
; 4320 values (8640 bytes) ranging eastward from 0 deg 0' East 
; longitude to 359 deg 55' East longitude (since it represents the 
; North Pole, all possible longitudes still refer to a single 
; point, thus the first band has 4320 identical values of -4290 m). 
; The 8641st starts the latitude band for 89 deg 55' N, and so on. 
; There is NO record for the South Pole (elevation 2810 m.) 
;----------------------------------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;----------------------------------------------------------------------
; Combine an ocean-based color map with an ocean/land colormap, by
; combining the first num_ocean colors of the GMT_ocean color map, 
; and every other color of the OceanLakeLandSnow colormap (minus the 
; first two blue colors).
;----------------------------------------------------------------------
undef("read_ocean_land_colormap")
function read_ocean_land_colormap(num_ocean)
local cmap_ocn, cmap_lnd
begin
  cmap_ocn = read_colormap_file("GMT_ocean")
  cmap_lnd = read_colormap_file("OceanLakeLandSnow")
  newcmap = array_append_record(cmap_ocn(0:num_ocean-1,:),cmap_lnd(2::2,:),0)
  return(newcmap)
end

;----------------------------------------------------------------------
; Given min/max elevation value and the number of colors you want
; for ocean, this function creates a custom color map AND calculates 
; levels for both land and ocean, based on an elevation pivot point.
;
; The assumption is that the end ocean index location is one
; less than the begining of the land index location.
;----------------------------------------------------------------------
undef("calc_levels_and_colors")
function calc_levels_and_colors(wks,emin,emax,split_elev,num_ocean_values)
local start_ocean, ocean_range, land_range, olevels, llevels, nol, nll, clen
begin
  cmap = read_ocean_land_colormap(num_ocean_values)
  clen = dimsizes(cmap(:,0))

  start_ocean = 0
  end_ocean   = num_ocean_values-1
  start_land  = end_ocean+1
  ocean_range = end_ocean-start_ocean+1
  land_range  = clen-start_land+1
  olevels     = fspan(emin,split_elev,ocean_range)
  llevels     = fspan(split_elev,emax,land_range)
  nol         = dimsizes(olevels)
  nll         = dimsizes(llevels)
  levels      = new((nol-1)+(nll-2),float)
  levels(0:nol-2) = olevels(1:)
  levels(nol-1:)  = llevels(1:nll-2)
  return([/levels,cmap/])
end

;----------------------------------------------------------------------
; This function reads a binary file containing elevation data and
; generates the necessary lat/lon coordinate arrays for plotting later.
; The information on the binary file is provided at the beginning of
; this script.
;
; The binary file was downloaded from:
;     http://www.ngdc.noaa.gov/mgg/global/relief/ETOPO5/TOPO/ETOPO5/
;----------------------------------------------------------------------
undef("read_elev_data")
function read_elev_data(topo_file)
local nlat, nlon, topo_file, lat, lon
begin
;---Read data as a straight binary file
  nlat = 2160
  nlon = 4320
  setfileoption("bin","ReadByteOrder","BigEndian")
  elev = cbinread(topo_file,(/nlat,nlon/),"short")

;---Create 1D coordinate arrays
  lat       = fspan(90,-90,nlat)
  lon       = fspan(0,360,nlon)
  lat!0     = "lat"
  lon!0     = "lon"
  lat@units = "degrees_north"
  lon@units = "degrees_east"
  lat&lat   = lat
  lon&lon   = lon

;---Attach the coordinate arrays
  elev!0    = "lat"
  elev!1    = "lon"
  elev&lat  = lat
  elev&lon  = lon
 
  return(elev)
end

;----------------------------------------------------------------------
; This procedure draws a global 5' topographic map by contouring the 
; given elevation data.
;----------------------------------------------------------------------
undef("draw_topo_map")
procedure draw_topo_map(wks,elev,title)
local res, labels, nlevels
begin
;---Set some resources for contouring and mapping
  res                    = True

  res@gsnMaximize        = True     ; maximize plot in frame

  res@cnFillOn           = True     ; turn on contour fill
  res@cnFillMode         = "MeshFill"
  res@cnLinesOn          = False    ; turn off contour lines
  res@cnLineLabelsOn     = False    ; turn off line labels
  res@cnInfoLabelOn      = False    ; turn off info label
  res@lbBoxLinesOn       = False    ; turn off labelbar box lines

;---Calculate "nice" contour levels, and create a color map to match
  split_elev               = -62; -68    ; meters
  num_ocean_colors         = 43
  levels_and_colors        = calc_levels_and_colors(wks,min(elev),max(elev),split_elev,num_ocean_colors)

  res@cnLevelSelectionMode = "ExplicitLevels"
  res@cnLevels             = levels_and_colors[0]
  res@cnFillPalette        = levels_and_colors[1]
  res@gsnAddCyclic         = False           ; don't add longitude cyclic point

  res@mpFillOn                    = False
  res@pmTickMarkDisplayMode       = "Always"   ; Nicer map labels
  res@mpGeophysicalLineThicknessF = 2          ; Thicker map outlines
  res@tiMainString                = title      ; Main title

;---Generate our own labels for the labelbar
  nlevels           = dimsizes(res@cnLevels)
  labels            = new(nlevels+2,string)
  labels            = ""                 ; Blank out all but
  ii                = ind(res@cnLevels.eq.split_elev)+1
  labels(0)         = "" + min(elev)     ; First,
  labels(nlevels+1) = "" + max(elev)     ; last, and 
  labels(ii)         = "" + split_elev   ; middle labels

  res@lbLabelAutoStride = False            ; This will make sure every labelbar 
  res@lbLabelAlignment = "ExternalEdges"   ; can potentially be labeled.
  res@lbLabelStrings   = labels

  plot = gsn_csm_contour_map(wks,elev,res)
end

;----------------------------------------------------------------------
; Main code
;----------------------------------------------------------------------
begin
  wks = gsn_open_wks("png","topo") ; send graphics to PNG file
  topo_filename = "ETOPO5.DAT"
  elev = read_elev_data(topo_filename)
  draw_topo_map(wks,elev,topo_filename)
end