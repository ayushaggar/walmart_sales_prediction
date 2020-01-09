Walmart Store Sales forecasting

## Objective
1) Predict Walmart sales for each store and department in it and to predict which stores are affected and the extent of the impact due to holidays or markdowns (Promotional Events)
2) Predict weekly sales of the department for store
3) Make a Rest API

**Assumptions** -
markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas

**Output** :
1) This analysis has been done using neural network. 
2) Analytical tools used in project are python, scikit learn and keras. 
3) User interface is developed by Flask web application

## Tools use 
> Python 3

> Main Libraries Used -
1) Tensorflow
2) Scikit-learn
3) Numpy
4) Flask
5) Pandas
6) Keras

**** 

## Installing and Running

> Folders Used -
1) flask - For Flask Application
2) flask/model - trained and exported model 
3) result - prediction result
4) data - having data in csv


```sh
$ cd walmart_sales_prediction
$ pip install -r requirements.txt
``` 

```
For Running Flask Application
```sh
$ python flask/api.py
```
Use http://0.0.0.0:5000/ for web application

****

## Various Steps in approach are -

1) Data is from Kaggle

2) Using machine learning model, a sales model is made which will help in prediction of sales

3) Following features or parameters are used – Store, Department, Date, Holiday, Temperature, Fuel Price, Markdown Events, CPI, Unemployment

4) Pre-processing and normalization of data is done 

5) Keras model is trained on normalized data  

6) Validation accuracy is calculated and model is exported with test data sales output

**Constraints / Notes** ::
1) Website made where we can submit store, department and date value and if it is special holiday or not and the outcome will be weekly sales value

2) Web application is a user interface which can be easily use by managers for forecasting sales of given date

**** 