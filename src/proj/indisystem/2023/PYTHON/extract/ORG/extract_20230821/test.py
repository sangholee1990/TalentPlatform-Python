import xarray as xr
import cfgrib
import pygrib
#ds=xr.open_dataset('D:\myProject\gribDB\data\ECMWF\ecmwf_20180101_0000.grib', engine='cfgrib', filter_by_keys={'typeOfLevel': 'shorName', 'topLevel':2})
#ds=cfgrib.open_dataset('D:\myProject\gribDB\data\LDAPS\l015_v070_erlo_unis_h024.2023062918.gb2')
#print(ds)

selCol=['SWDOWN', 'SWDOWNC', 'GSW', 'SWDDNI', 'SWDDIF', 'U10', 'V10']
dbCol=['SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']

for selCol1 in zip(selCol,dbCol) :
    print(selCol1)