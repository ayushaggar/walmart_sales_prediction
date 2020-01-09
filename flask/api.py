from __future__ import print_function  # In python 2.7
import sys
import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
import pickle
import datetime
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        model = joblib.load('flask/model/walmart_model.pkl')
        df_store = pd.read_csv('data/stores.csv')
        df_store['Type'] = df_store['Type'].map({'A': 0, 'B': 1, 'C': 2})
        store = request.form.get('store')
        dept = request.form.get('dept')
        sales_day = request.form.get('sales_day')
        IsHoliday = request.form.get('IsHoliday')
        month = datetime.datetime.strptime(sales_day, "%Y-%m-%d").month
        df_store = df_store.loc[df_store['Store'] == int(store)]
        Temperature = 100.14
        Fuel_Price = 4.468
        MarkDown1 = 0
        MarkDown2 = 0
        MarkDown3 = 0
        MarkDown4 = 0
        MarkDown5 = 0
        CPI = 0
        Unemployment = 0
        if not store or not dept or not sales_day:
            return render_template('index.html', label="No result put details")

        # make prediction on new value
        new_value = [
            store,
            dept,
            IsHoliday,
            df_store['Type'].iloc[0],
            df_store['Size'].iloc[0],
            Temperature,
            Fuel_Price,
            MarkDown1,
            MarkDown2,
            MarkDown3,
            MarkDown4,
            MarkDown5,
            CPI,
            Unemployment,
            month]
        test_scaler = pickle.load(open('flask/model/test_scaler.sav', 'rb'))
        test_scaled = test_scaler.fit_transform([new_value])
        X_test = test_scaled[:, :]
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        predicted_sales = model.predict(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
        predicted_weekly_sales = np.concatenate(
            (X_test[:, :], predicted_sales), axis=1)
        train_scaler = pickle.load(open('flask/model/train_scaler.sav', 'rb'))
        predicted_weekly_sales = train_scaler.inverse_transform(
            predicted_weekly_sales)
        return render_template(
            'index.html', label='Weekly_Sales = ' + str(predicted_weekly_sales[0][-1]))


if __name__ == '__main__':
    # start api
    app.run(host='0.0.0.0', port=8125, debug=True)
