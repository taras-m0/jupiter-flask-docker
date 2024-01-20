from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import sklearn


app = Flask(__name__)

# filename = '/web_flask_data/crop/model.sav'
# model = pickle.load(open(filename, 'rb'))

filename = '/web_flask_data/crop/model_jl.sav'
model = joblib.load(open(filename, 'rb'))

@app.route("/index.html", methods=['GET', 'POST'])
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        X = np.array([
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]).reshape(1, -1)

        # var_1 = float(request.form['var_1'])
        # var_2 = float(request.form['var_3'])
        # var_3 = float(request.form['var_3'])
        # X = np.array([88, 44, 40, 25, 81, 5.8, 230]).reshape(1, -1)
        y_pred = model.predict(X)[0]
        return render_template('index.html', culture=y_pred)
    else:
        return render_template('index.html', culture="")

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
