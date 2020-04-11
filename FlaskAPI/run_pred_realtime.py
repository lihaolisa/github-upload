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
    cur_weather = get_realtime_weather()
    for i in range(len(cur_weather)):
        features[cur_weather.index[i]] = cur_weather.values[i]
    now = datetime.now()
    features['Month'] = now.month
    features['HOUR'] = now.hour
    features['DAY_OF_WEEK'] = now.weekday() # Monday is 0 and Sunday is 6. # TODO: confirm if this is consistent with model
    features.to_csv(dir_path + '/data/realtime_features.csv', index=False)

    # TODO: Confirm if these are the features to run model. May need value transformation
    X = features 
    
    model = pickle.load(open(os.path.join(dir_path, 'model.pkl'), 'rb'))
    # TODO: let model prediction run through. Temporarily using random numbers
    # X['Prob'] = model.predict_proba(X)
    X['Prob'] = np.random.uniform(size=len(X))
    X.to_csv(dir_path + '/data/prediction_result.csv', index=False)

    # to json
    processed_results = X[['SEGMENT_ID', 'START_LONGITUDE','START_LATITUDE','END_LONGITUDE','END_LATITUDE', 'Prob']].set_index('SEGMENT_ID')
    results_json = processed_results.to_json(orient='index')
    # see an example output format in __main__
    return results_json


def call_google_map(origin, destination):
    # reference: https://github.com/googlemaps/google-maps-services-python
    google_map_key = 'AIzaSyA723t8eXV4ZpJgaoXBncDXLrlXdzr4tTw'
    gmaps = googlemaps.Client(key=google_map_key)

    # Geocoding an address
    origin_geocode = gmaps.geocode(origin)
    print('Origin geocode: ' + origin_geocode)
    destination_geocode = gmaps.geocode(destination)
    print('Destination geocode: ' + destination_geocode)

    # Look up an address with reverse geocoding
    # reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

    # Request directions 
    now = datetime.now()
    # our model predicts crash prob on Chicago’s arterial streets (nonfreeway), so avoid highways
    # alternatives: If True, more than one route may be returned in the response.
    directions_result = gmaps.directions(origin, destination,
                                        mode="driving",
                                        avoid='Highways',
                                        alternatives=True
                                        departure_time=now)
    """                                
    # parse json to retrieve all lat-lng
    waypoints = data['routes'][0]['legs']

    lats = []
    longs = []
    google_count_lat_long = 0

    # find cluster of interest from google api route
    for leg in waypoints:
        for step in leg['steps']:
            start_loc = step['start_location']
            # print("lat: " + str(start_loc['lat']) + ", lng: " + str(start_loc['lng']))
            lats.append(start_loc['lat'])
            longs.append(start_loc['lng'])
            google_count_lat_long += 1

    lats = tuple(lats)
    longs = tuple(longs)
    print("total waypoints: " + str(google_count_lat_long))

    return lats, longs, google_count_lat_long
"""


def calc_distance(accident_dataset, lats, longs, google_count_lat_long):
    # load all cluster accident waypoints to check against proximity
    accident_point_counts = len(accident_dataset.index)

    # approximate radius of earth in km
    R = 6373.0
    new = accident_dataset.append([accident_dataset] * (google_count_lat_long - 1), ignore_index=True)  # repeat data frame (9746*waypoints_count) times
    lats_r = list(
        itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in lats))  # repeat 9746 times
    longs_r = list(itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in longs))

    # append
    new['lat2'] = np.radians(lats_r)
    new['long2'] = np.radians(longs_r)

    # cal radiun50m
    new['lat1'] = np.radians(new['Latitude'])
    new['long1'] = np.radians(new['Longitude'])
    new['dlon'] = new['long2'] - new['long1']
    new['dlat'] = new['lat2'] - new['lat1']

    new['a'] = np.sin(new['dlat'] / 2) ** 2 + np.cos(new['lat1']) * np.cos(new['lat2']) * np.sin(new['dlon'] / 2) ** 2
    new['distance'] = R * (2 * np.arctan2(np.sqrt(new['a']), np.sqrt(1 - new['a'])))

    return new


def pred_search_route_prob(origin, destination):
    # get route planning
    call_google_map(origin, destination)
    """
    lats, longs, google_count_lat_long = 

    # calculate distance between past accident points and route
    dist = calc_distance(accident_dataset, lats, longs, google_count_lat_long)

    # filter for past accident points with distance <50m - route cluster
    dat = dist[dist['distance'] < 0.050][['Longitude','Latitude','Day_of_Week','Local_Authority_(District)',
                                               '1st_Road_Class','1st_Road_Number','Speed_limit', 'Year','Cluster',
                                               'Day_of_year', 'Hour']]

    #if no cluster, exit
    if len(dat) == 0:
        return print(" Hooray! No accidents predicted in your route.")

    else:
        # filter for accident points in route cluster
        #dat = accident_dataset[accident_dataset['Cluster'].isin(list(clusters['Cluster']))]
        dat = dat.drop(columns=['Hour', 'Day_of_year', 'Day_of_Week', 'Year'], axis=0)
        dat['Hour'] = datetime_object.hour
        day_of_year = (datetime_object - datetime.datetime(datetime_object.year, 1, 1)).days + 1
        dat['Day_of_year'] = day_of_year
        day_of_week = datetime_object.date().weekday() + 1
        dat['Day_of_Week'] = day_of_week
        dat['Year'] = datetime_object.year

        #get weather prediction for unique cluster in past accident dataset
        ucluster = list(dat['Cluster'].unique())
        clusters = dat[dat['Cluster'].isin(ucluster)].drop_duplicates(subset='Cluster', keep='first')
        weather = call_darksky(clusters, darkskykey, tm)

        # merge with accident data - df with latlong and weather
        final_df = pd.merge(dat, weather, how='left', on=['Cluster'])
        final_df = final_df.drop(columns=['time', 'summary', 'icon', 'ozone'], axis=0)
        final_df = final_df[model_columns]

        #run model predicition
        processed_results = model_pred(final_df)

        final = {}
        final["accidents"] = processed_results

        return final
        """
if __name__ == "__main__":
    # pred_result_json = pred_all_segments_prob()  # to show heat map
    # example of pred_result_json:
    # {"1308": {"START_LONGITUDE":"-87.639487",
    #           "START_LATITUDE":"41.874238",
    #           "END_LONGITUDE":"-87.627611",
    #           "END_LATITUDE":"41.874452",
    #           "Prob":0.5883092195},
    #  "1309": {"START_LONGITUDE":"-87.639487",
    #             "START_LATITUDE":"41.874538",
    #             "END_LONGITUDE":"-87.647125",
    #             "END_LATITUDE":"41.874438",
    #             "Prob":0.012699349}}
    # TODO: need to call Google map to get intermediate points in order to draw lines?
    
    origin = 'Chicago Illuminating Company, 2110 S Wabash Ave, Chicago, IL 60616'
    destination = 'The Bridgeport Art Center, 1200 W 35th St, Chicago, IL 60609'
    pred_search_route_prob(origin, destination)  # search route by calling google map. return crash prob on each route (number of routes >= 1)

