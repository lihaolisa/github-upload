import os
import csv
import numpy as np
import pandas as pd
import math
from datetime import datetime

def process_data_select_cols(path_in, path_out, cols_used=None):
    content = []
    with open(path_in) as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        header = next(reader)
        if cols_used:
            idx_cols = [header.index(c) for c in cols_used]
            for line in reader:
                content.append(map(line.__getitem__, idx_cols))
        else:  # explore data
            print(header)
            cols_used = header
            count = 0
            for line in reader:
                count += 1
                if count == 500:
                    break
                content.append(line)
    f.close()

    df = pd.DataFrame(content, columns=cols_used)
    df.to_csv(path_out, index=False)
    return


def process_congestion_data(path_in1, path_in2, path_out, start_date):
    header = ['TIME', 'SEGMENT_ID', 'SPEED']
    out = open(path_out, 'w')
    out.write(','.join(header) + '\n')
    with open(path_in2) as f:  # 2011 - 2018 data
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        header = next(reader)
        idx_cols = [header.index(c) for c in ['TIME', 'SEGMENTID', 'SPEED']]
        for line in reader:
            content = list(map(line.__getitem__, idx_cols))
            if content[-1] == '-1':  # -1 means no speed estimate
                content[-1] = ''
            # sample record time: 01/16/2013 11:50:32 PM
            # drop records before crash data is available
            record_time = datetime.strptime(content[0], '%m/%d/%Y %I:%M:%S %p')
            if record_time >= start_date:
                out.write(','.join(content) + '\n')
            
    with open(path_in1) as f:  # 2018 - current data
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        header = next(reader)
        idx_cols = [header.index(c) for c in ['TIME', 'SEGMENT_ID', 'SPEED']]
        for line in reader:
            content = list(map(line.__getitem__, idx_cols))
            if content[-1] == '-1':  # -1 means no speed estimate
                content[-1] = ''
            out.write(','.join(content) + '\n')
            
    f.close()
    out.close()
    return


def process_crash_data(path_in, path_out):
    df = pd.read_csv(path_in, parse_dates=['CRASH_DATE'])
    df.sort_values(by='CRASH_DATE', inplace=True, ascending=False)
    start_date = df['CRASH_DATE'].values[-1]
    end_date = df['CRASH_DATE'].values[0]
    print('Crash data date range ' + str(start_date) + ' to '+ str(end_date))
    df['POSTED_SPEED_LIMIT'] = df['POSTED_SPEED_LIMIT'].clip(10, 75)  # floor at 10, cap at 75
    df['TRAFFIC_CONTROL_DEVICE'] = df['TRAFFIC_CONTROL_DEVICE'].apply(lambda x:
         'RAILROAD CROSSING' if x in ['OTHER RAILROAD CROSSING', 'RAILROAD CROSSING GATE', 'RR CROSSING SIGN'] else
         ('UNKNOWN' if x in ['OTHER', 'UNKNOWN'] else x))
    df['DEVICE_CONDITION'].replace({'OTHER': 'UNKNOWN', 'WORN REFLECTIVE MATERIAL': 'FUNCTIONING IMPROPERLY',
                                    'MISSING': 'NO CONTROLS'}, inplace=True)
    df['WEATHER_CONDITION'].replace({'OTHER': 'UNKNOWN', 'WORN REFLECTIVE MATERIAL': 'FUNCTIONING IMPROPERLY',
                                    'MISSING': 'NO CONTROLS'}, inplace=True)
    df['TRAFFICWAY_TYPE'] = df['TRAFFICWAY_TYPE'].apply(lambda x: 'UNKNOWN' if x in ['OTHER', 'UNKNOWN', 'NOT REPORTED'] else x)
    print(df['LANE_CNT'].unique())
    df.loc[df['LANE_CNT'] > 6, 'LANE_CNT'] = np.nan
    print(df['LANE_CNT'].unique())
    df['ROADWAY_SURFACE_COND'].replace({'OTHER': 'UNKNOWN'}, inplace=True)
    df['ROAD_DEFECT'].replace({'OTHER': 'UNKNOWN'}, inplace=True)
    df['INJURY_or_TOW'] = df['CRASH_TYPE'].map({'NO INJURY / DRIVE AWAY': 0, 'INJURY AND / OR TOW DUE TO CRASH': 1})
    df.drop(['CRASH_TYPE'], axis=1, inplace=True)
    df['WORK_ZONE_TYPE'].replace({'UNKNOWN': ''}, inplace=True)
    df.to_csv(path_out, index=False)
    return


def process_congestion_data_realtime(path_in, path_out, path_map):
    df = pd.read_csv(path_in)
    df.columns = [x.strip() for x in df.columns.values]
    df.rename(columns={'CURRENT_SPEED': 'SPEED', 'SEGMENTID': 'SEGMENT_ID'}, inplace=True)
    df.sort_values(by=['SEGMENT_ID'], inplace=True)
    df['SPEED'].replace({-1: ''}, inplace=True)  # -1 means no estimate
    df.loc[:, ['SEGMENT_ID', 'SPEED']].to_csv(path_out, index=False)
    df.drop('SPEED', axis=1).to_csv(path_map, index=False)
    return


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
    b = get_distance_btw_points(lat, lon, line['start_lat'], line['start_long'])
    c = get_distance_btw_points(lat, lon, line['end_lat'], line['end_long'])
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


def assign_crash_to_segment(path_in, segment_ID_map, path_out):
    df_id_map = pd.read_csv(segment_ID_map, usecols=['SEGMENT_ID', 'START_LONGITUDE', 'START_LATITUDE', 'END_LONGITUDE', 'END_LATITUDE']).set_index('SEGMENT_ID')
    df_id_map.columns = [col.lower().replace('itude', '') for col in df_id_map.columns.values]
    dict_id_map = df_id_map.to_dict('index')
    for v in dict_id_map.values():
        v['distance'] = get_distance_btw_points(v['start_lat'], v['start_long'], v['end_lat'], v['end_long'])

    out = open(path_out, 'w')
    writer = csv.writer(out, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
    with open(path_in) as f:  # 2011 - 2018 data
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        header = next(reader)
        header_out = header + ['SEGMENT_ID']
        writer.writerow(header_out)
    
        idx_cols = [header.index(c) for c in ['LATITUDE', 'LONGITUDE']]
        for line in reader:
            try:
                lat = float(line[idx_cols[0]].strip())
                lon = float(line[idx_cols[1]].strip())
                mapped_id = find_segment_id(lat, lon, dict_id_map)
                line.append(mapped_id)
                writer.writerow(line)
            except:
                pass
                # drop the crashes without location info
    return


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    # 1. crash file
    path_crash_file = "../data/traffic_crashes_historical_all.csv"
    path_crash_file_select_cols = "../data/select_cols_crashes_historical_all.csv"
    path_crash_file_cleaned = "../data/cleaned_crashes_historical_all.csv"
    crash_file_cols_used = ['CRASH_DATE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', 'LATITUDE', 'LONGITUDE',
                            'POSTED_SPEED_LIMIT', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
                            'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE', 'LANE_CNT', 'ALIGNMENT',
                            'ROADWAY_SURFACE_COND', 'ROAD_DEFECT', 'WORK_ZONE_TYPE',
                            'STREET_NO', 'STREET_DIRECTION', 'STREET_NAME',
                            'CRASH_TYPE', 'MOST_SEVERE_INJURY', 'INJURIES_TOTAL', 'PRIM_CONTRIBUTORY_CAUSE', 'BEAT_OF_OCCURRENCE']
    process_data_select_cols(path_crash_file, path_crash_file_select_cols, crash_file_cols_used)
    process_crash_data(path_crash_file_select_cols, path_crash_file_cleaned)

    # 2. realtime congestion estimates
    path_congest_realtime = "../data/congestion_estimates_by_segments_realtime.csv"
    path_congest_realtime_select_cols = "../data/select_cols_congestion_realtime.csv"
    path_congest_realtime_cleaned = "../data/cleaned_congestion_realtime.csv"
    path_segment_ID_map = "../data/segment_ID_map.csv"  # Generate segment ID information. This is static
    congest_realtime_cols_used = ['SEGMENTID', ' CURRENT_SPEED', 'STREET', 'DIRECTION', 'FROM_STREET', 'TO_STREET', 'LENGTH',
                                  ' STREET_HEADING', 'START_LONGITUDE', ' START_LATITUDE', 'END_LONGITUDE', ' END_LATITUDE']
    process_data_select_cols(path_congest_realtime, path_congest_realtime_select_cols, congest_realtime_cols_used)
    process_congestion_data_realtime(path_congest_realtime_select_cols, path_congest_realtime_cleaned, path_segment_ID_map)

    # 3. historical congestion estimates
    crash_data_start = datetime.strptime('3/4/2013', '%m/%d/%Y')   
    path_congest_file1 = "../data/congestion_estimates_by_segments_2018-Current.csv"
    path_congest_file2 = "../data/congestion_estimates_by_segments_2011-2018.csv"
    # path_congest_file1 = "../data/congestion_estimates_by_segments_2018-Current_small.csv"
    # path_congest_file2 = "../data/congestion_estimates_by_segments_2011-2018_small.csv"
    path_congest_file_cleaned = "../data/cleaned_congestion.csv"
    process_congestion_data(path_congest_file1, path_congest_file2, path_congest_file_cleaned, crash_data_start)

    # 4. assign crash records to segment ID
    path_crash_file = "../data/cleaned_crashes_historical_all.csv"
    # path_crash_file = "../data/cleaned_crashes_historical_sample.csv"
    # process_data_select_cols(path_crash_file_all, path_crash_file)  # generate sample
    path_segment_ID_map = "../data/segment_ID_map.csv"
    path_out = "../data/cleaned_crashes_historical_all_ID.csv"
    assign_crash_to_segment(path_crash_file, path_segment_ID_map, path_out)

    