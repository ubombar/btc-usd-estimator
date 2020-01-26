import csv
import datetime
import numpy as np 
import random

random.seed()

FLAG_REBUILD = True

def get_csv_dict(many_days=10):
    data = []
    size = 0

    with open("BTC-USD/BTC_USD_2018-04-05_2020-01-25-CoinDesk.csv") as file:
        csv_file = csv.reader(file)
        all_data = []

        for i, row in enumerate(csv_file):
            if i == 0: continue

            size += 1

            date_str, close_price, open_price, high_price, low_price = tuple(row[1:])
            all_data.append((date_str, [float(close_price), float(open_price), float(high_price), float(low_price)]))
    return all_data, size

def preprocess_training_data(end_date='2019-12-31', before_date='2018-04-05', num_of_days=100, number_of_examples=200, directory='BTC-USD/btc-usd-train.npy'):
    all_data, size = get_csv_dict()

    temp_data = []
    temp_size = 0

    for date, values in all_data:
        if date == end_date: break
        if date == before_date: continue

        temp_data.append(values)
        temp_size += 1
    
    training_data = []
    training_label = []

    for _ in range(number_of_examples):
        start_index = random.randint(0, temp_size - num_of_days - 1)
        feature_data = []

        for i in range(num_of_days):
            feature_data.append(temp_data[i + start_index])

        training_data.append(feature_data)
        training_label.append(temp_data[num_of_days + start_index])

    numpy_array_data = np.array(training_data, dtype=np.float32)
    numpy_array_label = np.array(training_label, dtype=np.float32)

    np.save(directory + '-data', numpy_array_data)
    np.save(directory + '-label', numpy_array_label)
    # print(temp_size)

if FLAG_REBUILD:
    preprocess_training_data(end_date='2019-12-31', before_date='2018-04-05', directory='BTC-USD/btc-usd-train', number_of_examples=200)
    preprocess_training_data(end_date='2020-01-01', before_date='2020-01-25', directory='BTC-USD/btc-usd-test', number_of_examples=30)
    print("Done!")