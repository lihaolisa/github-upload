import os
import csv
import numpy as np
import pandas as pd
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
                if count == 2000:
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
    df.loc[df['LANE_CNT'] > 12, 'LANE_CNT'] = ''
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
    # process_data_select_cols(path_crash_file, path_crash_file_select_cols, crash_file_cols_used)
    # process_crash_data(path_crash_file_select_cols, path_crash_file_cleaned)

    # 2. realtime congestion estimates
    path_congest_realtime = "../data/congestion_estimates_by_segments_realtime.csv"
    path_congest_realtime_select_cols = "../data/select_cols_congestion_realtime.csv"
    path_congest_realtime_cleaned = "../data/cleaned_congestion_realtime.csv"
    path_segment_ID_map = "../data/segment_ID_map.csv"  # Generate segment ID information. This is static
    congest_realtime_cols_used = ['SEGMENTID', ' CURRENT_SPEED', 'STREET', 'DIRECTION', 'FROM_STREET', 'TO_STREET', 'LENGTH',
                                  ' STREET_HEADING', 'START_LONGITUDE', ' START_LATITUDE', 'END_LONGITUDE', ' END_LATITUDE']
    # process_data_select_cols(path_congest_realtime, path_congest_realtime_select_cols, congest_realtime_cols_used)
    # process_congestion_data_realtime(path_congest_realtime_select_cols, path_congest_realtime_cleaned, path_segment_ID_map)

    # 3. historical congestion estimates
    crash_data_start = datetime.strptime('3/4/2013', '%m/%d/%Y')   
    path_congest_file1 = "../data/congestion_estimates_by_segments_2018-Current.csv"
    path_congest_file2 = "../data/congestion_estimates_by_segments_2011-2018.csv"
    # path_congest_file1 = "../data/congestion_estimates_by_segments_2018-Current_small.csv"
    # path_congest_file2 = "../data/congestion_estimates_by_segments_2011-2018_small.csv"
    path_congest_file_cleaned = "../data/cleaned_congestion.csv"
    process_congestion_data(path_congest_file1, path_congest_file2, path_congest_file_cleaned, crash_data_start)
    