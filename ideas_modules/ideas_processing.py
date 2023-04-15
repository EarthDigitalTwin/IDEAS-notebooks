from datetime import datetime
from typing import List
import numpy as np
import xarray as xr
import pandas as pd
import time
import requests

import matplotlib.pyplot as plt

# Default values
DEFAULT_BASEMAP_RANGE = {
    'llcrnrlon': -180,
    'llcrnrlat': -90,
    'urcrnrlon': 180,
    'urcrnrlat': 90
}
dt_format = "%Y-%m-%dT%H:%M:%SZ"
units = {
    'pm25': 'µg/m³',
    'O3': 'mol m-2',
    'SO2': 'mmol/m2',
    'NO2': 'umol/m2',
    'CO': 'ppb',
    'CH4': 'ppb'
}
url_in_situ            = 'https://ideas-digitaltwin.jpl.nasa.gov/insitu/1.0'
endpoint_in_situ       = 'query_data_doms'
start_index_in_situ    = 0
items_per_page_in_situ = 500


'''
IDEAS endpoint functions
''' 

def spatial_timeseries(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to timeSeriesSpark IDEAS endpoint
    '''
    url = '{}/timeSeriesSpark?ds={}&minLon={}&minLat={}&maxLon={}&maxLat={}&startTime={}&endTime={}&lowPassFilter=False'.\
        format(base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
               start_time.strftime(dt_format), end_time.strftime(dt_format))

    # Display some information about the job
    print('url\n', url)
    print()

    # Query IDEAS to compute the time averaged map
    print("Waiting for response from IDEAS...", end="")
    start = time.perf_counter()
    ts_json = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return prep_ts(ts_json)


# def temporal_variance(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.DataArray:
#     '''
#     Makes request to varianceSpark IDEAS endpoint
#     '''
#     params = {
#         'ds': dataset,
#         'minLon': bb['min_lon'],
#         'minLat': bb['min_lat'],
#         'maxLon': bb['max_lon'],
#         'maxLat': bb['max_lat'],
#         'startTime': start_time.strftime(dt_format),
#         'endTime': end_time.strftime(dt_format)
#     }

#     url = '{}/varianceSpark?ds={}&minLon={}&minLat={}&maxLon={}&maxLat={}&startTime={}&endTime={}'.\
#         format(base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
#                start_time.strftime(dt_format), end_time.strftime(dt_format))

#     # Display some information about the job
#     print('url\n', url)
#     print()

#     # Query IDEAS to compute the time averaged map
#     print("Waiting for response from IDEAS... ", end="")
#     start = time.perf_counter()
#     var_json = requests.get(url, params=params, verify=False).json()
#     print("took {} seconds".format(time.perf_counter() - start))
#     return prep_var(var_json)


def data_subsetting(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.DataArray:
    '''
    Makes request to datainbounds IDEAS endpoint
    '''
    url = '{}/datainbounds?ds={}&b={},{},{},{}&startTime={}&endTime={}&lowPassFilter=False'.format(
        base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
        start_time.strftime(dt_format), end_time.strftime(dt_format))

    print('url\n', url)
    print()

    print("Waiting for response from IDEAS...", end="")
    start = time.perf_counter()
    var_json = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return prep_data_in_bounds(var_json)


def max_min_map_spark(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to maxMinMapSpark endpoint
    '''
    url = f'{base_url}/maxMinMapSpark?ds={dataset}&' \
          f'b={bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}&' \
          f'startTime={start_time.strftime(dt_format)}&endTime={end_time.strftime(dt_format)}'

    print('url\n', url)
    print()

    print("Waiting for response from IDEAS... ", end="")
    start = time.perf_counter()
    resp = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return max_min_prep(resp)


def daily_diff(base_url: str, dataset: str, clim: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.Dataset:
    '''
    Makes request to dailydifferenceaverage_spark endpoint
    '''
    url = f'{base_url}/dailydifferenceaverage_spark?dataset={dataset}&' \
          f'climatology={clim}&' \
          f'b={bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}&' \
          f'startTime={start_time.strftime(dt_format)}&endTime={end_time.strftime(dt_format)}'

    print('url\n', url)
    print()

    print("Waiting for response from IDEAS... ", end="")
    start = time.perf_counter()
    resp = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return daily_diff_prep(resp)


def temporal_mean(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime) -> xr.DataArray:
    '''
    Makes request to timeAvgMapSpark endpoint
    '''
    url = f'{base_url}/timeAvgMapSpark?ds={dataset}&' \
          f'b={bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}&' \
          f'startTime={start_time.strftime(dt_format)}&endTime={end_time.strftime(dt_format)}'

    print('url\n', url)
    print()

    print("Waiting for response from IDEAS... ", end="")
    start = time.perf_counter()
    resp = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return temporal_mean_prep(resp)


def hofmoeller(base_url: str, dataset: str, bb: dict, start_time: datetime, end_time: datetime, dim: str = 'latitude') -> xr.Dataset:
    '''
    Makes request to either latitudeTimeHofMoellerSpark or longitudeTimeHofMoellerSpark endpoint
    '''
    url = f'{base_url}/{dim}TimeHofMoellerSpark?ds={dataset}&' \
          f'b={bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}&' \
          f'startTime={start_time.strftime(dt_format)}&endTime={end_time.strftime(dt_format)}'

    print('url\n', url)
    print()

    print("Waiting for response from IDEAS... ", end="")
    start = time.perf_counter()
    resp = requests.get(url, verify=False).json()
    print("took {} seconds".format(time.perf_counter() - start))
    return hofmoeller_prep(resp, dim)


def insitu(base_url: str, provider: str, project: str, bb: str, start_time: datetime, end_time: datetime, var: str) -> pd.DataFrame:
    results = []
    base_url = base_url.replace('/nexus', '')
    next_url = f'{base_url}/insitu/1.0/query_data_doms_custom_pagination?startIndex=0&itemsPerPage=10000&' \
        f'provider={provider}&project={project}&startTime={datetime.strftime(start_time, "%Y-%m-%dT%H:%M:%SZ")}&' \
        f'endTime={datetime.strftime(end_time, "%Y-%m-%dT%H:%M:%SZ")}&bbox={bb}&variable={var}'

    while next_url:
        print(next_url)
        res = requests.get(next_url)
        results.append(res.json())
        if 'next' in res.json().keys() and res.json()['next'] != next_url:
            next_url = res.json()['next']
        else:
            break
    return prep_insitu(results)


'''
IDEAS endpoint response processing
'''


def prep_insitu(results: List) -> pd.DataFrame:
    all_results = []
    for r in results:
        if 'results' in r.keys():
            all_results.extend(r['results'])
    df = pd.DataFrame(all_results)
    return df


def prep_ts(ts_json: dict) -> xr.Dataset:
    '''
    Formats timeseriesspark response into xarray dataset object
    '''
    time = np.array([np.datetime64(ts[0]["iso_time"][:10])
                    for ts in ts_json["data"]])
    means = np.array([ts[0]["mean"] for ts in ts_json["data"]])
    mins = np.array([ts[0]["min"] for ts in ts_json["data"]])
    maxs = np.array([ts[0]["max"] for ts in ts_json["data"]])

    mean_da = xr.DataArray(means, coords=[time], dims=['time'], name='mean')
    min_da = xr.DataArray(mins, coords=[time], dims=['time'], name='minimum')
    max_da = xr.DataArray(maxs, coords=[time], dims=['time'], name='maximum')
    ds = xr.merge([mean_da, min_da, max_da])

    return ds


def prep_var(var_json: dict) -> xr.DataArray:
    '''
    Formats variancespark response into xarray dataarray object
    '''
    shortname = var_json['meta']['shortName']

    vals = np.array([v['variance'] for var in var_json['data'] for v in var])
    lats = np.array([var[0]['lat'] for var in var_json['data']])
    lons = np.array([v['lon'] for v in var_json['data'][0]])

    vals[vals == -9999] = np.nan

    vals_2d = np.reshape(
        vals, (len(var_json['data']), len(var_json['data'][0])))

    da = xr.DataArray(
        vals_2d, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])
    da.attrs['shortname'] = shortname
    da.attrs['units'] = '$m^2/s^2$'
    return da


def prep_data_in_bounds(var_json: dict) -> xr.DataArray:
    '''
    Formats datainbounds response into xarray dataarray object
    '''
    lat = []
    lon = []
    time = []

    for data in var_json:
        if data['latitude'] not in lat:
            lat.append(data['latitude'])
        if data['longitude'] not in lon:
            lon.append(data['longitude'])
        if data['time'] not in time:
            time.append(data["time"])

    lat.sort()
    lon.sort()
    time.sort()

    da = xr.DataArray(
        data=np.zeros((len(time), len(lat), len(lon))),
        dims=['time', 'lat', 'lon'],
        coords=dict(
            time=(['time'], time),
            lat=(['lat'], lat),
            lon=(['lon'], lon)
        )
    )

    for data in var_json:
        da.loc[data['time'], data['latitude'],
               data['longitude']] = data['data'][0]['variable']

    return da


def max_min_prep(var_json: dict) -> xr.Dataset:
    '''
    Formats maxmin response into xarray dataset object
    '''
    shortname = var_json['meta']['shortName']
    maxima = np.array([v['maxima'] for var in var_json['data'] for v in var])
    minima = np.array([v['minima'] for var in var_json['data'] for v in var])
    lat = np.array([var[0]['lat'] for var in var_json['data']])
    lon = np.array([v['lon'] for v in var_json['data'][0]])

    maxima_2d = np.reshape(
        maxima, (len(var_json['data']), len(var_json['data'][0])))
    minima_2d = np.reshape(
        minima, (len(var_json['data']), len(var_json['data'][0])))

    ds = xr.Dataset(
        data_vars=dict(
            maxima=(['lat', 'lon'], maxima_2d),
            minima=(['lat', 'lon'], minima_2d)
        ),
        coords=dict(
            lat=('lat', lat),
            lon=('lon', lon)
        ),
        attrs=dict(
            shortname=shortname
        )
    )

    return ds


def daily_diff_prep(var_json: dict) -> xr.Dataset:
    '''
    Formats dailydifference response into xarray dataset object
    '''
    shortname = var_json['meta']['shortName']
    mean = np.array([v['mean'] for var in var_json['data'] for v in var])
    std = np.array([v['std'] for var in var_json['data'] for v in var])
    time = np.array([np.datetime64(v["time"], 's')
                    for var in var_json['data'] for v in var])

    ds = xr.Dataset(
        data_vars=dict(
            mean=('time', mean),
            std=('time', std)
        ),
        coords=dict(
            time=('time', time)
        ),
        attrs=dict(
            shortname=shortname
        )
    )

    return ds


def temporal_mean_prep(var_json: dict) -> xr.DataArray:
    '''
    Formats timeavgmap response into xarray dataarray object
    '''
    lat = []
    lon = []

    for row in var_json['data']:
        for data in row:
            if data['lat'] not in lat:
                lat.append(data['lat'])
            if data['lon'] not in lon:
                lon.append(data['lon'])

    lat.sort()
    lon.sort()

    da = xr.DataArray(
        data=np.zeros((len(lat), len(lon))),
        dims=['lat', 'lon'],
        coords=dict(
            lat=(['lat'], lat),
            lon=(['lon'], lon)
        )
    )

    for row in var_json['data']:
        for data in row:
            da.loc[data['lat'], data['lon']] = data['mean']
    da = da.where(da != -9999, np.nan)
    return da


def hofmoeller_prep(var_json: dict, dim: str) -> xr.Dataset:
    '''
    Formats hofmoeller response into xarray dataset object
    '''
    times = [np.datetime64(s['time'], 's') for s in var_json['data']]
    if dim == 'latitude':
        dim_short = 'lats'
    else:
        dim_short = 'lons'
    dims = [l[dim] for l in var_json['data'][0][dim_short]]
    means = [l['mean'] for s in var_json['data'] for l in s[dim_short]]
    stds = [l['std'] for s in var_json['data'] for l in s[dim_short]]
    maxs = [l['max'] for s in var_json['data'] for l in s[dim_short]]
    mins = [l['min'] for s in var_json['data'] for l in s[dim_short]]

    mean_2d = np.reshape(means, (len(times), len(dims)))
    std_2d = np.reshape(stds, (len(times), len(dims)))
    max_2d = np.reshape(maxs, (len(times), len(dims)))
    min_2d = np.reshape(mins, (len(times), len(dims)))

    ds = xr.Dataset(
        data_vars=dict(
            mean=(['time', dim_short[:-1]], mean_2d),
            std=(['time', dim_short[:-1]], std_2d),
            max=(['time', dim_short[:-1]], max_2d),
            min=(['time', dim_short[:-1]], min_2d)
        ),
        coords=dict(
            time=(['time'], times),
            dim=([dim_short[:-1]], dims)
        )
    )
    return ds

###############################

def data_in_bounds(base_url, dataset, bb, start_time, end_time):
    url = '{}/datainbounds?ds={}&b={},{},{},{}&startTime={}&endTime={}&lowPassFilter=False'.format(
        base_url, dataset, bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'],
        start_time.strftime(dt_format), end_time.strftime(dt_format))
    
    # Display some information about the job
    print(url); print()
    
    # Query SDAP to get data-in-bounds
    print("Waiting for response from SDAP...")
    start = time.perf_counter()
    var_json = requests.get(url, verify=False).json()
    print("Time series took {} seconds".format(time.perf_counter() - start))
    return prep_data_in_bounds(var_json)

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