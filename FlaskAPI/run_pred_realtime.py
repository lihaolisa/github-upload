import requests
import pandas as pd
import numpy as np
import itertools
import os
import pickle
from sodapy import Socrata
import json
from datetime import datetime, timedelta, timezone, date
import os
import logging
import googlemaps
import math

dir_path = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger('myapp')
hdlr = logging.FileHandler(os.path.join(dir_path,'/tmp/myapp.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)

def get_realtime_congestion():
    # reference: https://dev.socrata.com/foundry/data.cityofchicago.org/n4j6-wkkf
    chicago_traffic_token = 'UamJH5YpwtspjPKiyGBTPMmLf'
    client = Socrata("data.cityofchicago.org", chicago_traffic_token)
    # First 2000 results, returned as JSON from API / converted to Python list of dictionaries by sodapy.
    results = client.get("n4j6-wkkf", limit=2000)
    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)

    results_df.columns = map(str.upper, results_df.columns)
    # TODO: confirm these contain the used features. Empty speed is kept as -1.
    # cols_to_keep and features are in mapped sequence
    cols_to_keep = ['SEGMENTID',  '_TRAFFIC','_DIRECTION', '_LENGTH', '_STRHEADING',   'START_LON',      '_LIF_LAT',      '_LIT_LON',     '_LIT_LAT']
    features =     ['SEGMENT_ID', 'SPEED',    'DIRECTION', 'LENGTH',  'STREET_HEADING','START_LONGITUDE','START_LATITUDE','END_LONGITUDE','END_LATITUDE']

    results_df = results_df.loc[:, cols_to_keep]
    results_df.rename(columns=dict(zip(cols_to_keep, features)), inplace=True)
    results_df['SEGMENT_ID'] = results_df['SEGMENT_ID'].astype(int)
    results_df = results_df.sort_values(by='SEGMENT_ID')
    results_df.to_csv(dir_path + '/data/realtime_prediction.csv', index=False)
    return results_df


def download_weather_data(dates):   
    url = "http://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e&startDate=" 
    weather = []
    for date in dates:
        date = date.strftime('%Y%m%d')
        res = requests.get(url + date)
        json_str = res.content
        logger.error("url:" + str(url + date))
        logger.error("json_str:" + str(json_str))
        new_dict = json.loads(json_str)
        try:
            weather.extend(new_dict['observations'])
        except:
            pass # at midnight, no new observations. use the last observation yesterday
    weather_df = pd.DataFrame(weather)
    return weather_df


def get_realtime_weather():
    today = date.today()
    yesterday = today - timedelta(days=1)
    keys_in_out_map = {'valid_time_gmt': 'Time',
                        'temp': 'Temperature (F)',
                        'dewPt': 'Dew Point (F)', 
                        'rh': 'Humidity (%)',
                        'wdir_cardinal': 'Wind Direction',
                        'wspd': 'Wind Speed (mph)',
                        'gust': 'Wind Gust (mph)',
                        'pressure': 'Pressure (in)',
                        'precip_hrly': 'Precipitation Hourly (in)',
                        'wx_phrase': 'Condition'
                    }

    weather_df = download_weather_data([yesterday, today])
    weather_df = weather_df.loc[:, keys_in_out_map.keys()]
    weather_df.rename(columns=keys_in_out_map, inplace=True)
    # convert UTC time to Pacific time then to Chicago time
    weather_df['Time'] = weather_df['Time'].apply(lambda x: (datetime.fromtimestamp(x, timezone.utc).astimezone() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"))
    weather_df = weather_df.sort_values(by='Time', ascending=False)
    weather_df.to_csv(dir_path + '/data/realtime_weather.csv', index=False)
    # only need the top row as the latest weather
    return weather_df.iloc[0, 1:]


def pred_all_segments_prob():
    features = get_realtime_congestion()
    features = features.astype({k: float for k in ['START_LONGITUDE','START_LATITUDE','END_LONGITUDE','END_LATITUDE']})
    cur_weather = get_realtime_weather()
    for i in range(len(cur_weather)):
        features[cur_weather.index[i]] = cur_weather.values[i]
    now = datetime.now()
    features['Month'] = now.month
    features['HOUR'] = now.hour
    features['DAY_OF_WEEK'] = now.weekday() # Monday is 0 and Sunday is 6. # TODO: confirm if this is consistent with model
    features.to_csv(dir_path + '/data/realtime_features.csv', index=False)

    # TODO: Confirm if these are the features to run model. May need value transformation
    X = features.loc[:, :] 
    
    model = pickle.load(open(os.path.join(dir_path, 'model.pkl'), 'rb'))
    # TODO: let model prediction run through. Temporarily using random numbers
    # features['Prob'] = model.predict_proba(X)
    features['Prob'] = np.random.uniform(size=len(X))
    features.to_csv(dir_path + '/data/prediction_result.csv', index=False)

    # to json
    processed_results = features[['SEGMENT_ID', 'START_LONGITUDE','START_LATITUDE','END_LONGITUDE','END_LATITUDE', 'Prob']].set_index('SEGMENT_ID')
    results_json = processed_results.to_json(orient='index')
    # see an example output format in __main__
    processed_results.to_json(orient='index', indent=4, path_or_buf=dir_path + '/data/prediction_result.json')
    return results_json


def call_google_map(origin, destination, multi_routes):
    # get routes from google map
    # reference: https://github.com/googlemaps/google-maps-services-python
    google_map_key = 'AIzaSyA723t8eXV4ZpJgaoXBncDXLrlXdzr4tTw'
    gmaps = googlemaps.Client(key=google_map_key)

    # Request directions 
    now = datetime.now()
    # our model predicts crash prob on Chicago’s arterial streets (nonfreeway), so avoid highways
    # alternatives: If True, more than one route may be returned in the response.
    directions_result = gmaps.directions(origin, destination,
                                        mode="driving",
                                        avoid='Highways',
                                        alternatives=multi_routes,
                                        departure_time=now)
    
    json_out = open(dir_path + '/data/google_map_result.json', 'w')
    json.dump(directions_result, json_out, indent=4, sort_keys=True)

    return directions_result


def parse_directions(directions):
    routes_info = []
    for route in directions:
        info = dict()
        info['distance'] = route['legs'][0]['distance']  # can show on UI
        info['duration'] = route['legs'][0]['duration']
        info['points_location'] = [] # list of dictionary, each step's lat lng
        for step in route['legs'][0]['steps']:
            info['points_location'].append(step['start_location']) 
        info['points_location'].append(route['legs'][0]['end_location'])  # add the end point
        routes_info.append(info)

    json_out = open(dir_path + '/data/parsed_routes_info.json', 'w')
    json.dump(routes_info, json_out, indent=4, sort_keys=True)
    return routes_info


def rad(d):
    return d * math.pi / 180.0

def get_distance_btw_points(lat1, lng1, lat2, lng2):
    # https://blog.csdn.net/tonkeytong/java/article/details/51445440
    EARTH_RADIUS = 6378.137  # in km
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2),2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2),2)))
    s = s * EARTH_RADIUS
    return s


def get_distance_point_to_line(lat, lon, line):
    # https://blog.csdn.net/ufoxiong21/article/details/46487001, section 2
    a = line['distance']
    b = get_distance_btw_points(lat, lon, line['START_LATITUDE'], line['START_LONGITUDE'])
    c = get_distance_btw_points(lat, lon, line['END_LATITUDE'], line['END_LATITUDE'])
    if b*b >= c*c + a*a:
        return c
    if c*c >= b*b + a*a:
        return b
    l = (a + b + c) / 2     # 周长的一半   
    s = math.sqrt(l*(l-a)*(l-b)*(l-c))  # 海伦公式求面积 
    return 2*s/a


def find_segment_id(lat, lon, dict_id_map):
    closest_id = np.nan
    min_dist = math.inf
    for key, value in dict_id_map.items():  # find segment id with min distance
        cur_dist = get_distance_point_to_line(lat, lon, value)
        if cur_dist < min_dist:
            min_dist = cur_dist
            closest_id = key
    return closest_id


def find_routes_segments(routes, segment_info):
    # get length of segment
    for v in segment_info.values():
        v['distance'] = get_distance_btw_points(v['START_LATITUDE'], v['START_LONGITUDE'], v['END_LATITUDE'], v['END_LONGITUDE'])
    json_out = open(dir_path + '/data/segment_info.json', 'w')
    json.dump(segment_info, json_out, indent=4, sort_keys=True)
    
    # assign points to segments
    for i in range(len(routes)):
        segments = set()
        for point in routes[i]['points_location']:
            closest_segment = find_segment_id(point['lat'], point['lng'], segment_info)
            if closest_segment not in segments:
                segments.add(closest_segment)
        routes[i]['segment_IDs'] = list(segments)

    json_out = open(dir_path + '/data/parsed_routes_info.json', 'w')
    json.dump(routes, json_out, indent=4, sort_keys=True)
    return routes


def pred_search_route_prob(origin, destination, pred_result_json):
    segment_info = json.loads(pred_result_json)  # load as dict: segment ID, lat lgn, prob
    segment_info = {int(k): v for k, v in segment_info.items()}
    
    # get route planning
    directions_result = call_google_map(origin, destination, multi_routes=True)
    # parse directions_result
    routes_info = parse_directions(directions_result)
    # use step locations on the routes, find segment IDs
    routes_info = find_routes_segments(routes_info, segment_info)

    # predict route crash prob
    for route in routes_info:
        prob_no_crash = 1
        for seg in route['segment_IDs']:
            seg_crash = segment_info[seg]['Prob']
            prob_no_crash *= (1 - seg_crash)
        route['crash prob'] = 1 - prob_no_crash
    
    json_out = open(dir_path + '/data/parsed_routes_info.json', 'w')
    json.dump(routes_info, json_out, indent=4, sort_keys=True)
    return json.dumps(routes_info)
    

if __name__ == "__main__":
    pred_result_json = pred_all_segments_prob()  # to show heat map
    # example of pred_result_json: see prediction_result.json
    # TODO: need to call Google map to get intermediate points in order to draw lines?
    
    origin = 'Chicago Illuminating Company, 2110 S Wabash Ave, Chicago, IL 60616'
    destination = 'The Bridgeport Art Center, 1200 W 35th St, Chicago, IL 60609'
    routes_info_json = pred_search_route_prob(origin, destination, pred_result_json)  # search route by calling google map. return crash prob on each route (number of routes >= 1)
    # example of routes_info_json: see parsed_routes_info.json
