from flask.helpers import url_for
from flask.templating import render_template_string
import numpy as np
import pandas as pd
from flask import Flask, app, request, render_template, redirect
import joblib

app = Flask(__name__)
aj = joblib.load('model_joblib')

@app.route('/', methods=['GET','POST'])
def home():
    if request.method=='POST':
        arr = list()
        result = request.form.values()
        for r in result:
            arr.append(float(r))
        arr = np.array(arr).reshape(1, -1)
        print(arr.shape)

        value = aj.predict(arr)

        return redirect(url_for('predict', val=value))
    return render_template('index.html')


@app.route('/predict/<val>')
def predict(val):
    val = val[1]
    print(val)
    print(type(val))
    if val == "1":
        result = "The person has breast cancer."
    else:
        result = "the person has not breast cancer."
    print(result)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)