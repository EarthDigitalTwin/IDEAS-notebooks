import warnings
warnings.filterwarnings("ignore")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
import matplotlib.colors as colors

import time 
import xarray as xr
import numpy as np
from typing import List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import textwrap
from datetime import datetime, timedelta

from tabulate import tabulate
from shapely.geometry import box

from IPython.display import Image
from owslib.util import Authentication
from owslib.wms import WebMapService
from PIL import Image as I
from PIL import ImageDraw


def timeseries_plot(data: List[Tuple[xr.DataArray, str]], x_label: str, y_label: str, title='', norm=False):
    '''
    Plots timeseries data on a chart
    '''

    plt.figure(figsize=(12, 5))

    for entry in data:
        da = entry[0]
        label = entry[1]
        if norm:
            vals = da.values / np.sqrt(np.sum(da.values**2))
        else:
            vals = da.values
        if len(entry) == 3:
            plt.plot(da.time, vals, linewidth=2,
                     label=textwrap.fill(label, 50), color=entry[2])
        else:
            plt.plot(da.time, vals, linewidth=2,
                     label=textwrap.fill(label, 50))

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    locator = mdates.DayLocator(interval=len(da.time)//8)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()


def plot_insitu(data: List[Tuple[pd.DataFrame, str, str]], title: str):
    fig = plt.figure(figsize=(12, 5))

    for df, var, label in data:
        if var == 'Streamflow':
            var_data = df[var]/35.315
        else:
            var_data = df[var]
        plt.plot(df.time, var_data, label=label)

    # plt.grid()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    # plt.xlabel(x_label, fontsize=12)
    plt.ylabel('m3/s', fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend(prop={'size': 12})


def base_map(bounds: dict = {}, padding: float = 2.5) -> plt.axes:
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

    countries = cf.NaturalEarthFeature(
        category='cultural', name='admin_0_countries', scale='10m', facecolor='none')
    rivers = cf.NaturalEarthFeature(
        category='physical', name='rivers_lake_centerlines', scale='10m', facecolor='none', edgecolor='blue')
    ax.add_feature(countries, zorder=100)
    ax.add_feature(rivers, zorder=101)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',
                      alpha=0.25, linestyle='--', draw_labels=True, zorder=90)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def map_box(bb: dict, padding=20):
    '''
    Adds bounding box to map
    '''
    ax = base_map(bb, padding)
    poly = Polygon([(bb['min_lon'], bb['min_lat']), (bb['min_lon'], bb['max_lat']), (bb['max_lon'], bb['max_lat']), (bb['max_lon'], bb['min_lat'])],
                   facecolor=(0, 0, 0, 0.0), edgecolor='red', linewidth=2, zorder=200)
    ax.add_patch(poly)
    plt.show()


def map_points(points: List, title='', zoom=False):
    '''
    Plots lat lon points on map
    points: list of tuples (lat, lon, label)
    '''
    ax = base_map()

    for (lat, lon, label) in points:
        ax.scatter([lon], [lat], s=50, alpha=1, label=label)

    ax.set_title(title)

    ax.set_xlim(-95, -86)
    ax.set_ylim(29, 35)

    if zoom:
        ax.set_xlim(-91.25, -90.75)
        ax.set_ylim(32.1, 32.75)

    ax.legend().set_zorder(102)


def map_data(data: xr.DataArray, title: str, cmap='rainbow', cb_label='', log_scale=False):
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
    if log_scale:
        mesh = ax.pcolormesh(x, y, data.values, norm=colors.LogNorm(), cmap=cmap, alpha=0.75)
    else:
        mesh = ax.pcolormesh(x, y, data.values, vmin=np.nanmin(data.values),
                            vmax=np.nanmax(data.values), cmap=cmap, alpha=0.75)
    cb = plt.colorbar(mesh)
    cb.set_label(cb_label)
    plt.title(title)
    plt.show()


def heatmap(data: xr.DataArray, x_label: str, y_label: str, title='', cmap='rainbow'):
    '''
    Plots colormesh heatmap
    '''
    time = [np.datetime_as_string(t, unit='D') for t in data.time]

    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(time, data.dim, data, cmap=cmap)
    plt.colorbar(mesh)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    locator = mdates.DayLocator(interval=5)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()


# Generalized notebook functions

def spatial_mean(base_url, dataset, bb, start_time, end_time, proc=[]):
    url = '{}/timeSeriesSpark?ds={}&minLon={}&minLat={}&maxLon={}&maxLat={}&startTime={}&endTime={}&lowPassFilter=False&seasonalFilter=False'.\
        format(base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
               start_time.strftime(dt_format), end_time.strftime(dt_format))

    # Display some information about the job
    print(url); print()

    # Query SDAP to compute the time averaged map
    print("Waiting for response from SDAP...")
    start = time.perf_counter()
    ts_json = requests.get(url, verify=False).json()
    print("Time series took {} seconds".format(time.perf_counter() - start))
    return prep_ts(ts_json, proc)

def prep_ts(ts_json, proc):
    shortname = ts_json['meta'][0]['shortName']
    time = np.array([np.datetime64(ts[0]["iso_time"][:10]) for ts in ts_json["data"]])
    vals = np.array([ts[0]["mean"] for ts in ts_json["data"]])
    
    da = xr.DataArray(vals, coords = [time], dims=['time'])
    
    for proc in proc:
        da = proc(da)
        
    da.attrs['shortname'] = shortname
        
    return da

def calc_anoms(data):
    return data - np.nanmean(data)

def comparison_plot(data, x_label, y_label, var='', anoms=False):
    plt.figure(figsize=(15,6))
    
    for da in data:
        if anoms:
            vals = calc_anoms(da.values)
        else:
            vals = da.values
        plt.plot(da.time, vals, linewidth=2, label=da.attrs['shortname'])
    
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel (y_label, fontsize=12)
    plt.xticks(rotation=45)
    plt.title(f'{var}{" Anomalies" if anoms else ""}', fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()
    
def temporal_variance(base_url, dataset, bb, start_time, end_time):
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(dt_format),
        'endTime': end_time.strftime(dt_format)
    }
    
    url = '{}/varianceSpark?ds={}&minLon={}&minLat={}&maxLon={}&maxLat={}&startTime={}&endTime={}'.\
        format(base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
               start_time.strftime(dt_format), end_time.strftime(dt_format))
    
    # Display some information about the job
    print(url); print()
    
    # Query SDAP to compute the time averaged map
    print("Waiting for response from SDAP...")
    start = time.perf_counter()
    var_json = requests.get(url, params=params, verify=False).json()
    print("Time series took {} seconds".format(time.perf_counter() - start))
    return prep_var(var_json)
    
def prep_var(var_json):
    shortname = var_json['meta']['shortName']

    vals = np.array([v['variance'] for var in var_json['data'] for v in var])
    lats = np.array([var[0]['lat'] for var in var_json['data']])
    lons = np.array([v['lon'] for v in var_json['data'][0]])
    
    vals[vals==-9999]=np.nan
    
    vals_2d = np.reshape(vals, (len(var_json['data']), len(var_json['data'][0])))

    da = xr.DataArray(vals_2d, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])
    da.attrs['shortname'] = shortname
    da.attrs['units'] = '$m^2/s^2$'
    return da

def get_in_situ_data(start_time: str, end_time: str,
                     min_lon: int, max_lon: int, min_lat: int, max_lat: int, provider: str) -> pd.DataFrame:
    data = []
    query_url = f'{url_in_situ}/{endpoint_in_situ}?' \
                f'startIndex={start_index_in_situ}&itemsPerPage={items_per_page_in_situ}&' \
                f'startTime={start_time}&endTime={end_time}&' \
                f'bbox={min_lon},{min_lat},{max_lon},{max_lat}&' \
                f'provider={provider}'
    if query_url:
        print(query_url)
    while query_url:
        resp = requests.get(query_url, verify=False).json()
        if len(resp['results']):
            data += resp['results']
        query_url = resp['next'].replace('http:', 'https:') if resp['last'] != resp['next'] else None

    
    return pd.DataFrame(data) if len(data) else None

def stacked_overlay_plot(x_datas: List[np.array], y_datas: List[np.array],
                series_labels: List[str], y_labels=List[str], title: str='',
                top_paddings: List[int]=[0, 0]):

    plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 1, sharex=True)

    # Plot 1
    ax[0].set_title(title)
    ax[0].plot(
        [ datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[0] ],
        y_datas[0], label=series_labels[0])
        
    # Plot 2
    ax[0].plot(
        [ datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[1] ],
        y_datas[1], label=series_labels[1])

    ax[0].legend(loc='upper center', shadow=True)
    y_data_max = max( np.amax(y_datas[0]), np.amax(y_datas[1]) )
    ax[0].set_ylim([ 0, y_data_max + top_paddings[0] ])
    ax[0].set_ylabel(y_labels[0])

    # Plot 3
    ax[1].plot(
        [ datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[2] ],
        y_datas[2], label=series_labels[2])
        
    # Plot 4
    ax[1].plot(
        [ datetime.strptime(x_val, '%Y-%m-%dT%H:%M:%SZ').replace(year=2022) for x_val in x_datas[3] ],
        y_datas[3], label=series_labels[3])
    
    ax[1].legend(loc='upper center', shadow=True)
    y_data_max = max(np.amax(y_datas[2]), np.amax(y_datas[3]))
    ax[1].set_ylim([ 0, y_data_max + top_paddings[1] ])
    ax[1].set_ylabel(y_labels[1])

    # Set title and legend
    plt.legend(loc='upper center', shadow=True)

    # Set grid and ticks
    dtFmt = mdates.DateFormatter('%b %d')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.xticks(rotation=45)
    ax[0].tick_params(left=False, bottom=False)
    ax[1].tick_params(left=False, bottom=False)
    ax[0].grid(b=True, which='major', color='k', linestyle='--', linewidth=0.25)
    ax[1].grid(b=True, which='major', color='k', linestyle='--', linewidth=0.25)
    
    plt.show()