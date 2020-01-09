from sys import argv

import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop

import pickle


def train_model(features, stores, train_data, test_data):
    df_features = pd.read_csv(features)
    df_store = pd.read_csv(stores)
    df_store['Type'] = df_store['Type'].map({'A': 0, 'B': 1, 'C': 2})
    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)

    # Make datetypes constant for all datasets
    df_features['Date'] = pd.to_datetime(
        df_features['Date'], format="%Y-%m-%d")
    df_train['Date'] = pd.to_datetime(df_train['Date'], format="%Y-%m-%d")
    df_test['Date'] = pd.to_datetime(df_test['Date'], format="%Y-%m-%d")

    # combine store data with train and test data
    df_train = pd.merge(df_train, df_store, how="left", on="Store")
    df_test = pd.merge(df_test, df_store, how="left", on="Store")

    # Drop  IsHoliday_y from feature data_set
    df_features = df_features.drop(["IsHoliday"], axis=1)

    # combine feature data with train and test data
    df_train = pd.merge(
        df_train, df_features, how="inner", on=[
            "Store", "Date"])
    df_test = pd.merge(df_test, df_features, how="inner", on=["Store", "Date"])

    # data pre processing
    processed_train = df_train.fillna(0)
    processed_test = df_test.fillna(0)

    # Replace -ve sales and markdown values by 0
    columns_process = [
        'Weekly_Sales',
        'MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5']
    for column_name in columns_process:
        processed_train.loc[processed_train[column_name]
                            < 0.0, column_name] = 0.0

    # Replace -ve sales and markdown values by 0
    columns_process = [
        'MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5']
    for column_name in columns_process:
        processed_test.loc[processed_test[column_name]
                           < 0.0, column_name] = 0.0

    # encoding for categorical and boolean data
    le = preprocessing.LabelEncoder()
    le.fit(processed_train['IsHoliday'].values.astype('str'))
    processed_train['IsHoliday'] = le.transform(
        processed_train['IsHoliday'].values.astype('str'))
    le.fit(processed_test['IsHoliday'].values.astype('str'))
    processed_test['IsHoliday'] = le.transform(
        processed_test['IsHoliday'].values.astype('str'))

    # put month data
    processed_train['month'] = pd.to_datetime(processed_train['Date']).dt.month
    processed_test['month'] = pd.to_datetime(processed_test['Date']).dt.month
    # processed_train = processed_train.drop(columns=["CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

    # put Weekly_Sales at last in tain data so as to use ML model
    cols_at_end = ['Weekly_Sales']
    processed_train = processed_train[[c for c in processed_train if c not in cols_at_end]
                                      + [c for c in cols_at_end if c in processed_train]]

    train_set = processed_train
    # print (train_set.max(axis = 0))
    test_set = processed_test
    # print (test_set.max(axis = 0))
    # Transforms features except date between 0 and 1
    train_set = train_set.set_index('Date')
    train_set_columns = list(train_set.columns)
    train_set_array = train_set.iloc[:, :].values
    train_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = train_scaler.fit_transform(train_set_array[:, :])
    test_set = test_set.set_index('Date')
    test_set_array = test_set.iloc[:, :].values
    test_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    test_set_scaled = test_scaler.fit_transform(test_set_array[:, :])

    # get prediction column in y train
    X_train, y_train = train_set_scaled[:, :-1], train_set_scaled[:, -1]
    X_test = test_set_scaled[:, :]
    # Create data structure

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    print (X_train.shape[2])
    # Initialising RNN
    model = Sequential()
    # Adding the first LSTM layer
    model.add(
        LSTM(
            units=10,
            return_sequences=True,
            activation='relu',
            input_shape=(
                X_train.shape[1],
                X_train.shape[2])))
    # Dropout regularization is added to avoid overfitting
    model.add(Dropout(0.5))
    model.add(LSTM(units=10, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=10, return_sequences=False, activation='relu'))
    model.add(Dropout(0.5))

    # Adding the output layer
    model.add(Dense(units=1, activation='sigmoid'))
    # Compiling the RNN
    model.compile(optimizer=RMSprop(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Fitting the RNN to the training set
    walmart_history = model.fit(X_train,
                                y_train,
                                epochs=1,
                                batch_size=128,
                                validation_split=0.2,
                                verbose=1)

    print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" %
          (100 * walmart_history.history['acc'][-1], 100 * walmart_history.history['val_acc'][-1]))

    model.save('flask/model/walmart_model.h5')
    with open('flask/model/walmart_model.pkl', 'wb') as fid:
        pickle.dump(model, fid)

    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    predicted_sales = model.predict(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    predicted_weekly_sales = np.concatenate(
        (X_test[:, :], predicted_sales), axis=1)
    predicted_weekly_sales = train_scaler.inverse_transform(
        predicted_weekly_sales)
    prediction = pd.DataFrame(
        predicted_weekly_sales,
        columns=train_set_columns)
    prediction = prediction.drop(['IsHoliday',
                                  'Type',
                                  'Size',
                                  'Temperature',
                                  'Fuel_Price',
                                  'MarkDown1',
                                  'MarkDown2',
                                  'MarkDown3',
                                  'MarkDown4',
                                  'MarkDown5',
                                  'CPI',
                                  'Unemployment',
                                  'month'],
                                 axis=1)
    prediction['Date'] = processed_test['Date']
    prediction['ID'] = processed_test['Store'].astype(
        str) + '_' + processed_test['Dept'].astype(str) + '_' + processed_test['Date'].astype(str)
    prediction = prediction.drop(['Store', 'Dept', 'Date'], axis=1)
    cols = prediction.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    prediction = prediction[cols]
    prediction.to_csv('result/prediction_result.csv', index=False)
    pickle.dump(train_scaler, open('flask/model/train_scaler.sav', 'wb'))
    pickle.dump(test_scaler, open('flask/model/test_scaler.sav', 'wb'))


def main():
    if len(argv) != 5:
        print("Please provide only 4 argument which should be the input file name and save file name")
        return
    features = argv[1]
    stores = argv[2]
    train_data = argv[3]
    test_data = argv[4]
    train_model(features, stores, train_data, test_data)


if __name__ == "__main__":
    main()
else:
    print ("Executed when imported")
