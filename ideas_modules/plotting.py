from typing import List, Tuple
import xarray as xr
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import numpy as np
import textwrap


def timeseries_plot(data: List[Tuple[xr.DataArray, str]], x_label: str, y_label: str, title='', norm=False):
    '''
    Plots timeseries data on a chart
    '''

    plt.figure(figsize=(12,5))
    
    for da, label in data:
        if norm:
            vals = da.values / np.sqrt(np.sum(da.values**2))
        else:
            vals = da.values
        plt.plot(da.time, vals, linewidth=2, label=textwrap.fill(label, 50))
    
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel (y_label, fontsize=12)
    locator = mdate.DayLocator(interval=len(da.time)//8)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()

def base_map(bounds:dict = {}, padding:float=2.5) -> plt.axes:
    '''
    Creates map with bounds and padding
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if bounds:
        bounds = (bounds['min_lon'] - padding,
                bounds['max_lon'] + padding,
                bounds['min_lat'] - padding,
                bounds['max_lat'] + padding)
    else:
        bounds = (-180, 180, -90, 90)
    ax.set_extent(bounds, ccrs.PlateCarree())

    ax.add_feature(cf.LAND)
    ax.add_feature(cf.OCEAN)
    ax.coastlines('10m')
    ax.add_feature(cf.STATES, zorder=100)

    countries = cf.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', facecolor='none')
    rivers = cf.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines', scale='10m', facecolor='none', edgecolor='blue')
    ax.add_feature(countries, zorder=100)
    ax.add_feature(rivers, zorder=101)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.25, linestyle='--', draw_labels=True, zorder=90)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax

def map_box(bb: dict):
    '''
    Adds bounding box to map
    '''
    ax = base_map(bb, padding=20)
    poly = Polygon([(bb['min_lon'],bb['min_lat']),(bb['min_lon'],bb['max_lat']),(bb['max_lon'],bb['max_lat']),(bb['max_lon'],bb['min_lat'])],
                   facecolor=(0,0,0,0.0),edgecolor='red',linewidth=2,zorder = 200)
    ax.add_patch(poly)
    plt.show()
    
def map_data(data: xr.DataArray, title: str, cmap='rainbow'):
    '''
    Plots data on map
    '''
    bounds = {
        'min_lon': data.lon.min(),
        'max_lon': data.lon.max(),
        'min_lat': data.lat.min(),
        'max_lat': data.lat.max()
    }
    ax = base_map(bounds)
    x, y = np.meshgrid(data.lon, data.lat)
    mesh = ax.pcolormesh(x, y, data.values, vmin=np.nanmin(data.values),
                         vmax=np.nanmax(data.values), cmap=cmap, alpha=0.75)
    plt.colorbar(mesh)
    plt.title(title)
    plt.show()

def heatmap(data: xr.DataArray, x_label: str, y_label: str, title='', cmap='rainbow'):
    '''
    Plots colormesh heatmap
    '''
    plt.figure(figsize=(12,5))
    mesh = plt.pcolormesh(data.time, data.dim, data, cmap=cmap)
    plt.colorbar(mesh)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    locator = mdate.DayLocator(interval=5)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()