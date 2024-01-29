import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as mp
import matplotlib.pyplot as plt

class DrawSatelliteImage :
    def __init__(self,data,lat,lon) :
        self.data=data
        self.lat=lat
        self.lon=lon

    def run(self) :
        fig = plt.figure(figsize=(16,12))
#draw basemap on fig
        m = Basemap(projection='lcc',
                    resolution='f',
                    lat_0=35.5,
                    lon_0=126,
                     llcrnrlat=32.97114181518555,
                     llcrnrlon=124.09576416015625,
                     urcrnrlat=38.83988571166992,
                     urcrnrlon=130.054931640625)
        m.drawcoastlines()
        x,y= m(self.lon, self.lat)
#        my_cmap = plt.get_cmap('rainbow')
#        my_cmap.set_under('white')

        #cs = m.pcolor(x,y,self.data,shading='flat',edgecolors='none',cmap=plt.cm.jet)
        m.scatter(x, y, marker='D',color='m')
        plt.colorbar()

        print(x.shape)
#        print(y)
#        print(self.data)
        plt.show()
        """"""