import xarray as xr
import Nio
#import matplotlib.pyplot as plt
fname="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_unis_h024.2023062918.gb2"

#ds = xr.open_dataset(fname,engine="pynio")
#xc = ds.coords

#print(xc)


f= Nio.open_file(fname,"r")
fc=f.dimensions.keys()
fv=f.variables.keys()
"""
print("===================================================")
coord_nio = f.dimensions.keys()
varNames  = f.variables.keys()

for i in varNames:
    print(f.variables[i])
    print(f.variables[i][:])

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
