# -*- coding: utf-8 -*-
"""

"""

import requests
import json
from collections import defaultdict
import time
with open('data/prediction_result.json') as f:
    data=json.load(f)

new_data=defaultdict()
GEONAMES_USER = "birdking"
GEONAMES_API = "http://api.geonames.org/"
nearintersection_url = "{}findNearestIntersectionJSON?lat={{}}&lng={{}}&username={}".format(GEONAMES_API, GEONAMES_USER)
def get_response_writedic (lng,lat,desc,idx):
    url = nearintersection_url.format(lat, lng)
    response = requests.get(url)
    raw_result = json.loads(response.text)
    if (raw_result['credits']=='1.0'):
        try:
            new_data[idx][str.upper(desc)+'_LONGITUDE']=float(raw_result['intersection']['lng'])
            new_data[idx][str.upper(desc)+'_LATITUDE']=float(raw_result['intersection']['lat'])
        except:
            new_data[idx][str.upper(desc)+'_LONGITUDE']=float(data[idx][str.upper(desc)+'_LONGITUDE'])
            new_data[idx][str.upper(desc)+'_LATITUDE']=float(data[idx][str.upper(desc)+'_LATITUDE'])
    
for i in data.keys():
    new_data[i]=defaultdict()
    get_response_writedic(data[i]['START_LONGITUDE'],data[i]['START_LATITUDE'],'START',i)
    get_response_writedic(data[i]['END_LONGITUDE'],data[i]['END_LATITUDE'],'END',i)
    new_data[i]['Prob']=float(data[i]['Prob'])
    new_data[i]['distance']=float(data[i]['distance'])

with open('output{}.json'.format(int(time.time())), 'w') as json_file:
    json.dump(new_data,json_file)