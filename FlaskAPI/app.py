from flask import Flask, request, jsonify, render_template
from run_pred_realtime import pred_all_segments_prob, pred_search_route_prob
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plot_heatmap')
def plot_heatmap():  # this is used to visualize heatmap
    data_pred = json.loads(pred_all_segments_prob())
    # To assign probability to corrected coordinates
    with open('data/coRef.json') as f:
        coord_ref = json.load(f)
    for i in coord_ref:
        coord_ref[i]['Prob']=data_pred[i]['Prob']
    return render_template('heatmap.html',data_list=coord_ref)

@app.route('/route_plan')
def route_plan():
    return render_template('route_plan_map.html')

@app.route('/call_search_route', methods=['POST'])
def call_search_route():  # how this is used - see route_plan_map.html, url = "{{ url_for('call_search_route') }}";
    request_data = request.get_json()
    origin = request_data['origin']
    destination = request_data['destination']

    routes_info_json = pred_search_route_prob(origin, destination)
    return routes_info_json

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)