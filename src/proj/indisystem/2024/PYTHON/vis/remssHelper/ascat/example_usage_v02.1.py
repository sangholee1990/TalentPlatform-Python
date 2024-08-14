from ascat_daily import ASCATDaily

def read_data(filename='ascat_20071022_v02.1.gz'):
    dataset = ASCATDaily(filename, missing=missing)
    if not dataset.variables: sys.exit('file not found')
    return dataset

lons = (36,46)
lats = (-26,-16)
iasc = 1
wspdname = 'windspd'
wdirname = 'winddir'
missing = -999.

def myvmax(): return 20
def myscale(): return 320
def mycolor(): return 'black'

#----------------------------------------------------------------------------

def show_dimensions(ds):
    print('')
    print('Dimensions')
    for dim in ds.dimensions:
        aline = ' '.join([' '*3, dim, ':', str(ds.dimensions[dim])])
        print(aline)

def show_variables(ds):
    print('')
    print('Variables:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':', ds.variables[var].long_name])
        print(aline)

def show_validrange(ds):
    print('')
    print('Valid min and max and units:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':',
                str(ds.variables[var].valid_min), 'to',
                str(ds.variables[var].valid_max),
                '(',ds.variables[var].units,')'])
        print(aline)

def set_image(vmin,vmax,extent):
    myimage = {}
    myimage['origin'] = 'lower' 
    myimage['vmin'] = vmin
    myimage['vmax'] = vmax
    myimage['extent'] = extent
    myimage['interpolation'] = 'nearest'
    return myimage

def quikquiv(plt,lon,lat,u,v,scale,region,color):    
    # selecting the sub-region is not necessary,
    # but it greatly reduces time needed to render plot   
    ilon1,ilon2,ilat1,ilat2 = region
    xx = lon[ilon1:ilon2+1]
    yy = lat[ilat1:ilat2+1]
    uu = u[ilat1:ilat2+1,ilon1:ilon2+1]
    vv = v[ilat1:ilat2+1,ilon1:ilon2+1]
    plt.quiver(xx,yy,uu,vv,scale=scale,color=color) 

def show_plotexample(dataset, figname='plot_example.png'):
    print('')
    print('Plot example:')

    # modules needed for this example:
    import numpy as np
    import pylab as plt
    from matplotlib import cm

    # here is the data I will use:
    wspd = dataset.variables[wspdname][iasc,:,:]
    wdir = dataset.variables[wdirname][iasc,:,:]
    land = dataset.variables['land'][iasc,:,:]

    # get lon/lat:
    lon = dataset.variables['longitude']
    lat = dataset.variables['latitude']

    # get metadata:
    name = dataset.variables[wspdname].long_name
    units = dataset.variables[wspdname].units
    vmin = dataset.variables[wspdname].valid_min
    vmax = dataset.variables[wspdname].valid_max

    # get extent of dataset:    
    extent = []
    extent.append(dataset.variables['longitude'].valid_min)
    extent.append(dataset.variables['longitude'].valid_max)
    extent.append(dataset.variables['latitude'].valid_min)
    extent.append(dataset.variables['latitude'].valid_max)

    # get region to plot:   
    ilon1 = np.argmin(np.abs(lons[0]-lon))
    ilon2 = np.argmin(np.abs(lons[1]-lon))
    ilat1 = np.argmin(np.abs(lats[0]-lat))
    ilat2 = np.argmin(np.abs(lats[1]-lat))
    region = (ilon1,ilon2,ilat1,ilat2)

    # get u and v from wspd and wdir:
    from bytemaps import get_uv
    u,v = get_uv(wspd,wdir)
    bad = np.where(wspd<0)
    u[bad] = 0.
    v[bad] = 0.

    # set colors:
    palette = cm.jet
    palette.set_under('black')
    palette.set_over('grey')
    wspd[land] = 1.E30

    # my preferences:
    vmax = myvmax()
    scale = myscale()
    color = mycolor()

    # make the plot:
    fig = plt.figure()
    plt.imshow(wspd,**set_image(vmin,vmax,extent))
    plt.colorbar()
    plt.xlim(lons)
    plt.ylim(lats)
    quikquiv(plt,lon,lat,u,v,scale,region,color)
    plt.title(name+' ('+units+')')
    plt.grid()    
    fig.savefig(figname)
    print(' '.join([' '*3,'Saving:',figname]))

#----------------------------------------------------------------------------

if __name__ == '__main__':    
    import sys
    dataset = read_data()
    show_dimensions(dataset)
    show_variables(dataset)
    show_validrange(dataset)
    show_plotexample(dataset)
    print('')
    print('done')
    print('')
