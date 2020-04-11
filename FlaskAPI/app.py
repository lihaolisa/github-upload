# refer to sample project - CODE/app/main/routes.py
import numpy as np
from flask import Flask, request, jsonify, render_template
from  run_pred_realtime import pred_all_segments_prob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heatmap',methods=['POST'])
def heatmap():

    pred_result_json = pred_all_segments_prob()

    return render_template('index.html', prediction_text='The result is {}'.format('Success!'))

@app.route('/route_plan',methods=['POST'])
def route_plan():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)