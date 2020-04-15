from flask import Flask, request, jsonify, render_template
from run_pred_realtime import pred_all_segments_prob, pred_search_route_prob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/call_segment_prob', methods=['POST'])
def call_segment_prob():  # this is used to visualize heatmap
    pred_result_json = pred_all_segments_prob()
    return pred_result_json

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