import xarray as xr
import Nio
#import matplotlib.pyplot as plt
#fname="/vol01/DATA/MODEL/KIM/r030_v040_ne36_unis_h001.2023063000.gb2"
#fname="/vol01/DATA/MODEL/KIM/r030_v040_ne36_pres_h006.2023063000.gb2"
#fname="/vol01/DATA/MODEL/ECMWF/ecmwf_20180101_0000.grib"
fname="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_unis_h024.2023062918.gb2"
#fname="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_pres_h024.2023062918.gb2"

#ds = xr.open_dataset(fname,engine="pynio")
#xc = ds.coords

#print(xc)


f= Nio.open_file(fname,"r")

print("===================================================")
fc=f.dimensions.keys()
fv=f.variables.keys()
fd = f.dimensions
#xxx=f.variables['RH_P0_L100_GLC0'].attributes.keys()
#print(xxx)
#data=f.variables['CDIR_GDS0_SFC'][:]
#data1=f.variables['forecast_time0'][:]
#print(data1)
print(fc)
print(fd)
print(fv)
#print(data)
#print(data.shape)

#for i in fv:
#    print(f.variables[i])
#    print(f.variables[i][:])

"""
print("===================================================")
gridlon_0= f.variables['gridlon_0'][:]
dimensions_nio = f.dimensions
shape_nio = gridlon_0.shape
size_nio  = gridlon_0.size
rank_nio  = len(shape_nio)   # or rank_nio = f.variables["tsurf"].rank

print('dimensions: ', dimensions_nio)
print('shape:      ', shape_nio)
print('size:       ', size_nio)
print('rank_nio:   ', rank_nio)

print("===================================================")
attributes_nio = list(f.variables['gridlon_0'].attributes.keys())

print('attributes_nio: ', attributes_nio)

print("===================================================")

long_name_nio = f.variables["gridlon_0"].attributes['long_name']
units_nio = f.variables["gridlon_0"].attributes['units']

print('long_name_nio: ', long_name_nio)
print('units_nio:     ', units_nio)
"""
