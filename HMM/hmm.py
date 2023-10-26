from tqdm import tqdm
import itertools
from hmmlearn.hmm import GaussianHMM
import io
import requests
import time
import datetime
import yfinance as yf
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = yf.download("AAPL", start="2010-01-01", end="2023-10-01")
data.head()
data.shape
train_size = int(0.8*data.shape[0])
print(train_size)
train_data = data.iloc[0:train_size]
test_data = data.iloc[train_size+1:]


def augment_features(dataframe):
    fracocp = (dataframe['Close']-dataframe['Open'])/dataframe['Open']
    frachp = (dataframe['High']-dataframe['Open'])/dataframe['Open']
    fraclp = (dataframe['Open']-dataframe['Low'])/dataframe['Open']
    new_dataframe = pd.DataFrame({'delOpenClose': fracocp,
                                 'delHighOpen': frachp,
                                  'delLowOpen': fraclp})
    new_dataframe.set_index(dataframe.index)

    return new_dataframe


def extract_features(dataframe):
    return np.column_stack((dataframe['delOpenClose'], dataframe['delHighOpen'], dataframe['delLowOpen']))


features = extract_features(augment_features(train_data))
features.shape
model = GaussianHMM(n_components=10)
feature_train_data = augment_features(train_data)
features_train = extract_features(feature_train_data)
model.fit(features_train)

test_augmented = augment_features(test_data)
fracocp = test_augmented['delOpenClose']
frachp = test_augmented['delHighOpen']
fraclp = test_augmented['delLowOpen']

sample_space_fracocp = np.linspace(fracocp.min(), fracocp.max(), 50)
sample_space_fraclp = np.linspace(fraclp.min(), frachp.max(), 10)
sample_space_frachp = np.linspace(frachp.min(), frachp.max(), 10)

possible_outcomes = np.array(list(itertools.product(
    sample_space_fracocp, sample_space_frachp, sample_space_fraclp)))
num_latent_days = 50
num_days_to_predict = 300

predicted_close_prices = []
for i in tqdm(range(num_days_to_predict)):
    # Calculate start and end indices
    previous_data_start_index = max(0, i - num_latent_days)
    previous_data_end_index = max(0, i)
    # Acquire test data features for these days
    previous_data = extract_features(augment_features(
        test_data.iloc[previous_data_start_index:previous_data_end_index]))

    outcome_scores = []
    for outcome in possible_outcomes:
        # Append each outcome one by one with replacement to see which sequence generates the highest score
        total_data = np.row_stack((previous_data, outcome))
        outcome_scores.append(model.score(total_data))

    # Take the most probable outcome as the one with the highest score
    most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
    predicted_close_prices.append(
        test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0]))
    import matplotlib.pyplot as plt

plt.figure(figsize=(30, 10), dpi=80)
plt.rcParams.update({'font.size': 18})

x_axis = np.array(
    test_data.index[0:num_days_to_predict], dtype='datetime64[ms]')
plt.plot(x_axis, test_data.iloc[0:num_days_to_predict]
         ['Close'], 'b+-', label="Actual close prices")
plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
plt.legend(prop={'size': 20})
plt.show()
ae = abs(test_data.iloc[0:num_days_to_predict]
         ['Close'] - predicted_close_prices)

plt.figure(figsize=(30, 10), dpi=80)

plt.plot(x_axis, ae, 'go-', label="Error")
plt.legend(prop={'size': 20})
plt.show()
print("Max error observed = " + str(ae.max()))
print("Min error observed = " + str(ae.min()))
print("Mean error observed = " + str(ae.mean()))
num_latent_days_values = [10, 20, 30, 40, 50, 60]
baseline_num_latent_days = 50
n_components_values = [4, 6, 8, 10, 12, 14]
baseline_n_componets = 10
num_steps_values = [10, 20, 40, 50]
baseline_num_steps = 50
num_days_to_predict = 100  # We don't need to predict as many days as before
mae_num_components = []
for num_component in n_components_values:
    model = GaussianHMM(n_components=num_component)
    model.fit(features_train)
    predicted_close_prices = []
    for i in tqdm(range(num_days_to_predict)):
        # Calculate start and end indices
        previous_data_start_index = max(0, i - baseline_num_latent_days)
        previous_data_end_index = max(0, i)
        # Acquire test data features for these days
        previous_data = extract_features(augment_features(
            test_data.iloc[previous_data_start_index:previous_data_end_index]))

        outcome_scores = []
        for outcome in possible_outcomes:
            # Append each outcome one by one with replacement to see which sequence generates the highest score
            total_data = np.row_stack((previous_data, outcome))
            outcome_scores.append(model.score(total_data))

        # Take the most probable outcome as the one with the highest score
        most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
        predicted_close_prices.append(
            test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0]))
    mae_num_components.append(
        (abs(test_data.iloc[0:num_days_to_predict]['Close'] - predicted_close_prices)).mean())
plt.figure(figsize=(30, 10), dpi=80)

plt.plot(n_components_values, mae_num_components, 'go-', label="Error")
plt.xlabel("Number of hidden states")
plt.ylabel("MAE")
plt.legend(prop={'size': 20})
plt.show()
mae_num_steps = []
model = GaussianHMM(n_components=baseline_n_componets)
model.fit(features_train)
for num_step in num_steps_values:
    sample_space_fracocp = np.linspace(fracocp.min(), fracocp.max(), num_step)
    sample_space_fraclp = np.linspace(
        fraclp.min(), frachp.max(), int(num_step/5))
    sample_space_frachp = np.linspace(
        frachp.min(), frachp.max(), int(num_step/5))
    possible_outcomes = np.array(list(itertools.product(
        sample_space_fracocp, sample_space_frachp, sample_space_fraclp)))
    predicted_close_prices = []
    for i in tqdm(range(num_days_to_predict)):
        # Calculate start and end indices
        previous_data_start_index = max(0, i - baseline_num_latent_days)
        previous_data_end_index = max(0, i)
        # Acquire test data features for these days
        previous_data = extract_features(augment_features(
            test_data.iloc[previous_data_start_index:previous_data_end_index]))

        outcome_scores = []
        for outcome in possible_outcomes:
            # Append each outcome one by one with replacement to see which sequence generates the highest score
            total_data = np.row_stack((previous_data, outcome))
            outcome_scores.append(model.score(total_data))

        # Take the most probable outcome as the one with the highest score
        most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
        predicted_close_prices.append(
            test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0]))
    mae_num_steps.append(
        (abs(test_data.iloc[0:num_days_to_predict]['Close'] - predicted_close_prices)).mean())
plt.figure(figsize=(30, 10), dpi=80)

plt.plot(num_steps_values, mae_num_steps, 'go-', label="Error")
plt.xlabel("Number of intervals for features")
plt.ylabel("MAE")
plt.legend(prop={'size': 20})
plt.show()
