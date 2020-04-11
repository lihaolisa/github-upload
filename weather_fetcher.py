import json
import requests
from datetime import datetime, timedelta, timezone
import csv

# web link: https://www.wunderground.com/history/daily/us/il/chicago/KMDW/date/2019-4-3
# see "Daily Observation" table in the above link
url = "http://api.weather.com/v1/location/KMDW:9:US/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e&startDate="

def get_date_list(start,end):
    date_list= []
    date = datetime.strptime(start,'%Y%m%d')
    end = datetime.strptime(end,'%Y%m%d')
    while date <= end:
        date_list.append(date.strftime('%Y%m%d'))
        date = date + timedelta(1)
    return date_list


def download_data_for_date(date):    
    res = requests.get(url + date)
    json_str = res.content
    new_dict = json.loads(json_str)
    return new_dict['observations']


def get_weather_data(date_list):
    csv_data_file = open('../data/weather_data.csv', 'w') 
    csv_writer = csv.writer(csv_data_file)   
    count = 0
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

    for date in date_list:
        weather_date = download_data_for_date(date)
        for weather_hr in weather_date:
            hr_data = {}
            for key_in, key_out in keys_in_out_map.items():
                hr_data[key_out] = weather_hr[key_in]
            utc_time = datetime.fromtimestamp(hr_data['Time'], timezone.utc) # hr_data['Time'] is UTC time
            local_time = utc_time.astimezone() + timedelta(hours=2)  # convert to my local time (Pacific), then Chicago time
            hr_data['Time'] = local_time.strftime("%Y-%m-%d %H:%M:%S")
            if count == 0: 
                header = hr_data.keys() 
                csv_writer.writerow(header) 
                count += 1
            csv_writer.writerow(hr_data.values()) 
    
    csv_data_file.close() 


date_list = get_date_list('20130304','20200321')
get_weather_data(date_list)