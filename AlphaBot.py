#%% Imports

# General
from distutils.log import Log
from msilib import knownbits
from random import Random
import pandas as pd
import numpy as np
from math import *
import pandas_datareader.data as web
import datetime as dt
import pickle

import warnings

warnings.filterwarnings("ignore")

import datetime as dt

# Statistical
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsRegressor

# Custom
from PredictorPipeline import *


#%% Utility Functions

def clean_data(crypto_dataset, non_crypto_dataset, crypto_asset_list, non_crypto_asset_list, predicted_security = "BTC", predicted_field = "return",
                prediction_type = "regression", long_only_classif_threshold = None, absolute_threshold = None):

    crypto_dataset_clean = crypto_dataset.copy()
    crypto_dataset_clean["Date"] = pd.to_datetime(crypto_dataset_clean["Date"], format = "%d/%m/%Y %H:%M")
    crypto_dataset_clean["Date"] = crypto_dataset_clean["Date"].dt.date
    non_crypto_dataset["Date"] = non_crypto_dataset["Date"].dt.date

    combined_dataset = pd.merge(crypto_dataset_clean, non_crypto_dataset, how = 'left', on = "Date")
    combined_dataset = combined_dataset.sort_values(by = "Date")
    
    combined_dataset = combined_dataset.fillna(method = 'backfill')

    crypto_asset_list.extend(non_crypto_asset_list)
    asset_list = crypto_asset_list
    # Calculate composites:
    for asset in asset_list:
        # Return current period and return lag:
        combined_dataset[asset + "_close_lag"] = combined_dataset[asset + "_Close"].shift(1)
        combined_dataset[asset + "_return"] = (combined_dataset[asset + "_Close"]/combined_dataset[asset + "_close_lag"]) - 1
        combined_dataset[asset + "_return_lag"] = combined_dataset[asset + "_return"].shift(1)

        # Relativize volumes
        combined_dataset[asset + "_volume_lag"] = combined_dataset[asset + "_Volume"].shift(1)
        combined_dataset[asset + "_vol_growth"] = (combined_dataset[asset + "_Volume"]/combined_dataset[asset + "_volume_lag"]) - 1
        combined_dataset.drop(asset + "_volume_lag", axis = 1)

        # Calculate rolling volatility
        combined_dataset["5_" + asset + "_vol"] = combined_dataset[asset + "_return"].rolling(5).std()
        combined_dataset["20_" + asset + "_vol"] = combined_dataset[asset + "_return"].rolling(20).std()
        combined_dataset["60_" + asset + "_vol"] = combined_dataset[asset + "_return"].rolling(60).std()

    combined_dataset = combined_dataset.replace([np.inf, -np.inf], np.nan)

    # Specify which asset to predict
    if prediction_type == "regression":
        combined_dataset[predicted_security + "_period_ahead_" + predicted_field] = combined_dataset[predicted_security + "_" + predicted_field].shift(-1)
        combined_dataset = combined_dataset.rename(columns = {predicted_security + "_" + predicted_field: "lag"})
    elif prediction_type == "classification":
        if absolute_threshold != None:
            combined_dataset[predicted_security + "_period_ahead_" + predicted_field] = combined_dataset[predicted_security + "_" + predicted_field].shift(-1)
            combined_dataset.loc[combined_dataset[predicted_security + "_period_ahead_return"] > absolute_threshold, [predicted_security + "_period_ahead_" + predicted_field]] = 1
            combined_dataset.loc[combined_dataset[predicted_security + "_period_ahead_return"] <= absolute_threshold, [predicted_security + "_period_ahead_" + predicted_field]] = 0
        elif long_only_classif_threshold != None:
            combined_dataset[predicted_security + "_period_ahead_" + predicted_field] = combined_dataset[predicted_security + "_" + predicted_field].shift(-1)
            combined_dataset.loc[combined_dataset[predicted_security + "_period_ahead_" + predicted_field] > combined_dataset[predicted_security + "_period_ahead_" + predicted_field].quantile(long_only_classif_threshold), [predicted_security + "_period_ahead_" + predicted_field]] = 1
            combined_dataset.loc[combined_dataset[predicted_security + "_period_ahead_" + predicted_field] <= combined_dataset[predicted_security + "_period_ahead_" + predicted_field].quantile(long_only_classif_threshold), [predicted_security + "_period_ahead_" + predicted_field]] = 0

    combined_dataset = combined_dataset.dropna(thresh = len(combined_dataset)*0.8, axis = "columns")
    combined_dataset = combined_dataset.dropna()

    return combined_dataset

#%% Statistical Functions

class StrategySimulator(E2EModelDevelopment):

    def __init__(self, dataset, model, Y_name, date_name, bid_ask = 0, trade_commission = 0):

        super().__init__(dataset, model, Y_name, date_name)            

        self.bid_ask = bid_ask
        self.trade_commission = trade_commission

    @staticmethod
    def optimize_decision_threshold(pretrained_model, x_train, y_train):
        print("START")
        attempts = 0
        candidate_threshold = 0
        accuracy = 0 # starter values
        base_proportion = y_train.mean() # starter values
        while accuracy < base_proportion and attempts < 20 and candidate_threshold < 1:
            prob_pred = pretrained_model.predict_proba(x_train)[:,0]
            y_pred = np.where(prob_pred < candidate_threshold, 0, 1)
            
            tn, fp, fn, tp = confusion_matrix(y_pred = y_pred, y_true = y_train).ravel()
            accuracy = (tp + tn)/(tp + tn + fp + fn)

            candidate_threshold = candidate_threshold + 0.01
            print(accuracy)
            print(candidate_threshold)
            print(base_proportion)
            attempts = attempts + 1

        optimal_threshold = candidate_threshold

        return optimal_threshold
    
    @staticmethod
    def take_strategy_returns(direction, performance_data, trade_commission, bid_ask):

        if direction == "long_only":
            
            # performance_data["unique_signal_identifier"] =  (performance_data["prediction_binary"]*performance_data["prediction_binary_lag"]) + (performance_data["prediction_binary"] - performance_data["prediction_binary_lag"])
            performance_data["cost_adjusted_entry_price"] = performance_data["current_close"] + (performance_data["current_close"]*trade_commission) + (performance_data["current_close"]*bid_ask)
            performance_data["cost_adjusted_exit_price"] = performance_data["next_close"] - (performance_data["next_close"]*trade_commission) - (performance_data["next_close"]*bid_ask)

            for i in range(0,len(performance_data)):
                if performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = 0
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["next_close"].iloc[i]/performance_data["current_close"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["cost_adjusted_exit_price"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = 0
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = 0
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["cost_adjusted_exit_price"].iloc[i]/performance_data["current_close"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["next_close"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = 0

        elif direction == "long_short":
            
            # performance_data["unique_signal_identifier"] =  (performance_data["prediction_binary"]*performance_data["prediction_binary_lag"]) + (performance_data["prediction_binary"] - performance_data["prediction_binary_lag"])
            performance_data["cost_adjusted_entry_price"] = performance_data["current_close"] + (performance_data["current_close"]*trade_commission) + (performance_data["current_close"]*bid_ask)
            performance_data["cost_adjusted_exit_price"] = performance_data["next_close"] - (performance_data["next_close"]*trade_commission) - (performance_data["next_close"]*bid_ask)

            for i in range(0,len(performance_data)):
                if performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = -((performance_data["next_close"].iloc[i]/performance_data["current_close"].iloc[i]) - 1)
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["next_close"].iloc[i]/performance_data["current_close"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["cost_adjusted_exit_price"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = -((performance_data["next_close"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1)
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = -((performance_data["cost_adjusted_exit_price"].iloc[i]/performance_data["current_close"].iloc[i]) - 1)
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["cost_adjusted_exit_price"].iloc[i]/performance_data["current_close"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 0 and performance_data["prediction_binary"].iloc[i] == 1 and performance_data["prediction_binary_next"].iloc[i] == 1:
                    performance_data["strategy_returns"].iloc[i] = (performance_data["next_close"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1
                elif performance_data["prediction_binary_lag"].iloc[i] == 1 and performance_data["prediction_binary"].iloc[i] == 0 and performance_data["prediction_binary_next"].iloc[i] == 0:
                    performance_data["strategy_returns"].iloc[i] = -((performance_data["next_close"].iloc[i]/performance_data["cost_adjusted_entry_price"].iloc[i]) - 1)
        
        return performance_data


    def classif_strat(self, train_window, test_window, step = 1, normalize = False,
                    actual_returns_plugin = None, binary_rebalance = False, direction = "long_only", 
                    optimize_threshold = False, prediction_type = "classification"):
        
        splitter = TimeCrossValidation(input_data = self.OOS_dataset, 
                                train_window = train_window,
                                test_window = test_window, step = test_window, date_column =  self.date_name)
        train_test_split_index = splitter.rolling_window_train_test_split()

        strategy_performance = pd.DataFrame()
        dummy_strategy_performance = pd.DataFrame()

        OOS_cross_fold_y_pred = pd.DataFrame()
        OOS_cross_fold_y_actual = pd.DataFrame()

        dummy_cross_fold_y_pred = pd.DataFrame()
        dummy_cross_fold_y_actual = pd.DataFrame()
        
        for train_index, test_index in train_test_split_index:
            dataset_train = self.dataset[(self.dataset[self.date_name] >= train_index[0]) & (self.dataset[self.date_name] <= train_index[1])]
            
            imbalanced_X_train = dataset_train.drop([self.Y_name, self.date_name], axis = 1)
            imbalanced_Y_train = dataset_train[self.Y_name]

            if binary_rebalance:
                majority_class = dataset_train[self.Y_name].mode()[0]
                minority_class_filtered_dataset = dataset_train[dataset_train[self.Y_name] != majority_class]
                majority_class_filtered_dataset = dataset_train[dataset_train[self.Y_name] == majority_class]
                majority_class_filtered_dataset = majority_class_filtered_dataset.sample(n = len(minority_class_filtered_dataset), replace = True)
                dataset_train = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis = 0)

            dataset_val = self.dataset[(self.dataset[self.date_name] > test_index[0]) & (self.dataset[self.date_name] <= test_index[1])]
            X_train = dataset_train.drop([self.Y_name, self.date_name], axis = 1)
            X_val = dataset_val.drop([self.Y_name, self.date_name], axis = 1)
            Y_train = dataset_train[self.Y_name]
            Y_val = dataset_val[self.Y_name]

            if normalize:

                transformer = RobustScaler().fit(X_train)

                X_train_columns = X_train.columns
                X_train = transformer.transform(X_train)
                X_train = pd.DataFrame(X_train)
                X_train.columns = X_train_columns

                X_val_columns = X_val.columns
                X_val = transformer.transform(X_val)
                X_val = pd.DataFrame(X_val)
                X_val.columns = X_val_columns

            candidate_model = self.optimal_model

            candidate_model.fit(X_train, Y_train)

            if optimize_threshold:
                optimal_threshold = StrategySimulator.optimize_decision_threshold(candidate_model, imbalanced_X_train, imbalanced_Y_train)
                prob_pred = candidate_model.predict_proba(X_val)[:,0]
                y_pred = pd.DataFrame(np.where(prob_pred < optimal_threshold, 0, 1))
            else:
                y_pred = pd.DataFrame(candidate_model.predict(X_val))
            
            unaugmented_Y_val = Y_val.copy()

            Y_val = pd.merge(Y_val, actual_returns_plugin, how = "left", left_index = True, right_index=True)
            return_dataframe = pd.merge(y_pred.reset_index(drop = True), Y_val.reset_index(drop = True), left_index = True, right_index = True)
            return_dataframe.columns = ["prediction_binary", "actual_binary", "actual", "current_close", "next_close", "Date"]

            dummy_model = self.dummy_model
            dummy_y_pred = pd.DataFrame(dummy_model.predict(X_val))
            dummy_return_dataframe = pd.merge(dummy_y_pred.reset_index(drop = True), Y_val.reset_index(drop = True), left_index = True, right_index = True)
            dummy_return_dataframe.columns = ["prediction_binary", "actual_binary", "actual", "current_close", "next_close", "Date"]

            strategy_performance = strategy_performance.append(return_dataframe)
            dummy_strategy_performance = dummy_strategy_performance.append(dummy_return_dataframe)

            # Prediction performance
            OOS_cross_fold_y_pred = OOS_cross_fold_y_pred.append(pd.DataFrame(y_pred))
            OOS_cross_fold_y_actual = OOS_cross_fold_y_actual.append(pd.DataFrame(unaugmented_Y_val))

            # Dummy Model:
            if prediction_type == "classification":
                dummy_clf = DummyClassifier(strategy="stratified")
                self.dummy_model = dummy_clf
                dummy_clf.fit(X_train, Y_train)
                dummy_y_pred = dummy_clf.predict(X_val)
                dummy_cross_fold_y_pred = dummy_cross_fold_y_pred.append(pd.DataFrame(dummy_y_pred))
                dummy_cross_fold_y_actual = dummy_cross_fold_y_actual.append(pd.DataFrame(unaugmented_Y_val))


        strategy_performance["prediction_binary_lag"] = strategy_performance["prediction_binary"].shift(-1) # TODO: check that it's correct lag
        strategy_performance["prediction_binary_next"] = strategy_performance["prediction_binary"].shift(1) # TODO: check that it's correct lag
        strategy_performance = strategy_performance.dropna()
        strategy_performance["strategy_returns"] = None

        strategy_performance = StrategySimulator.take_strategy_returns(direction = direction, performance_data = strategy_performance, trade_commission = self.trade_commission, bid_ask = self.bid_ask)

        dummy_strategy_performance["prediction_binary_lag"] = dummy_strategy_performance["prediction_binary"].shift(-1) # TODO: check that it's correct lag
        dummy_strategy_performance["prediction_binary_next"] = dummy_strategy_performance["prediction_binary"].shift(1) # TODO: check that it's correct lag
        dummy_strategy_performance = dummy_strategy_performance.dropna()
        dummy_strategy_performance["strategy_returns"] = None

        dummy_strategy_performance = StrategySimulator.take_strategy_returns(direction = direction, performance_data = dummy_strategy_performance, trade_commission = self.trade_commission, bid_ask = self.bid_ask)

        # Prediction performance:
        OOS_cross_fold_y_pred = OOS_cross_fold_y_pred.reset_index(drop = True)
        OOS_cross_fold_y_actual = OOS_cross_fold_y_actual.reset_index(drop = True)

        if prediction_type == "classification":
            dummy_cross_fold_y_pred = dummy_cross_fold_y_pred.reset_index(drop = True)
            dummy_cross_fold_y_actual = dummy_cross_fold_y_actual.reset_index(drop = True)

        OOS_cross_fold_y_pred.columns = ["prediction"]
        if prediction_type == "classification":
            accuracy_dataframe = pd.DataFrame(OOS_cross_fold_y_pred.values == OOS_cross_fold_y_actual.values)
            OOS_final_eval_score = float(accuracy_dataframe.mean())

            dummy_accuracy_dataframe = pd.DataFrame(dummy_cross_fold_y_pred.values == dummy_cross_fold_y_actual.values)
            dummy_final_eval_score = float(dummy_accuracy_dataframe.mean())

        print("model and dummy scores: ", OOS_final_eval_score, dummy_final_eval_score)

        return strategy_performance, dummy_strategy_performance


#%% Import Data

# 1. Bitcoin Data:
BTCUSD = pd.read_csv('data\gemini_BTCUSD_day.csv')
BTCUSD = BTCUSD[["Date", "Close", "High", "Low", "Volume"]]
BTCUSD = BTCUSD.rename(columns = {"Close":"BTC_Close", "Volume":"BTC_Volume", 
                        "High":"BTC_High", "Low":"BTC_Low"})

ETHUSD = pd.read_csv('data\gemini_ETHUSD_day.csv')
ETHUSD = ETHUSD[["Date", "Close", "High", "Low", "Volume"]]
ETHUSD = ETHUSD.rename(columns = {"Close":"ETH_Close", "Volume":"ETH_Volume", 
                        "High":"ETH_High", "Low":"ETH_Low"})

LTCUSD = pd.read_csv('data\gemini_LTCUSD_day.csv')
LTCUSD = LTCUSD[["Date", "Close", "High", "Low", "Volume"]]
LTCUSD = LTCUSD.rename(columns = {"Close":"LTC_Close", "Volume":"LTC_Volume", 
                        "High":"LTC_High", "Low":"LTC_Low"})

ZECUSD = pd.read_csv('data\gemini_ZECUSD_day.csv')
ZECUSD = ZECUSD[["Date", "Close", "High", "Low", "Volume"]]
ZECUSD = ZECUSD.rename(columns = {"Close":"ZEC_Close", "Volume":"ZEC_Volume", 
                        "High":"ZEC_High", "Low":"ZEC_Low"})

# 2. Other Data
start = dt.datetime(2015,1,1)
end = dt.datetime(2022,1,15)

#'^GSPC', '^IXIC', '^RUT', '^IRX', '^TNX', 
list_of_assets = ['EURUSD=X', 'GBPUSD=X', 'GC=F']

asset_data_parent = web.DataReader("^VIX",'yahoo', start, end).reset_index()
asset_data_parent = asset_data_parent.drop("Adj Close", axis = 1)
asset_data_parent["Date"] = pd.to_datetime(asset_data_parent["Date"])
asset_data_parent = asset_data_parent.add_prefix("^VIX" + '_')
asset_data_parent = asset_data_parent.rename(columns = {('^VIX' + '_' + 'Date'):'Date'})
for asset in list_of_assets:
    asset_data = web.DataReader(asset,'yahoo', start, end).reset_index()
    asset_data = asset_data.drop("Adj Close", axis = 1)
    asset_data["Date"] = pd.to_datetime(asset_data["Date"])
    asset_data = asset_data.add_prefix(asset + '_')
    asset_data = asset_data.rename(columns = {(asset + '_' + 'Date'):'Date'})
    asset_data_parent = pd.merge(asset_data_parent, asset_data, how = "left", on = "Date")

asset_data_parent[asset_data_parent.columns] = asset_data_parent[asset_data_parent.columns].apply(pd.to_numeric, errors='ignore', axis=1)


# %% Process Data

# Join
crypto_dataset = pd.merge(BTCUSD, ETHUSD, how = "inner", on = "Date")
crypto_dataset = pd.merge(crypto_dataset, LTCUSD, how = "inner", on = "Date")
crypto_dataset = pd.merge(crypto_dataset, ZECUSD, how = "inner", on = "Date")

# Process
crypto_to_trade = "BTC"
asset_data_parent_ready = asset_data_parent.copy()
crypto_dataset_clean = clean_data(crypto_dataset = crypto_dataset, non_crypto_dataset = asset_data_parent_ready, crypto_asset_list = ["BTC", "LTC", "ETH", "ZEC"], non_crypto_asset_list = list_of_assets, predicted_security = crypto_to_trade, prediction_type = "classification", long_only_classif_threshold = None, absolute_threshold = 0)

except_date = list(crypto_dataset_clean.columns)
except_date.remove("Date")
crypto_dataset_clean[except_date] = crypto_dataset_clean[except_date].apply(pd.to_numeric, errors='coerce')

# %% Correlation of Crypto Assets

correlation_matrix = crypto_dataset_clean[["BTC_return", "ETH_return", "LTC_return", "ZEC_return"]].corr()
# crypto_dataset_clean[["BTC_return", "ETH_return", "LTC_return", "ZEC_return"]].plot()

print(correlation_matrix)

#%% Normality and Randomness Plots

crypto_dataset_clean["BTC_return"].hist(bins = 100)

crypto_dataset_clean[["BTC_return", "Date"]].plot(x = "Date", y = "BTC_return", fontsize=8)


#%% Correlation with Lags

crypto_dataset_clean[["BTC_period_ahead_return", "BTC_return", "LTC_return", "ETH_return", "ZEC_return"]].corr()

# %% Calculate Returns on Trading Strats

crypto_to_trade = "BTC"
asset_data_parent_ready = asset_data_parent.copy()
crypto_dataset_clean = clean_data(crypto_dataset = crypto_dataset, non_crypto_dataset = asset_data_parent_ready, crypto_asset_list = ["BTC", "LTC", "ETH", "ZEC"], non_crypto_asset_list = list_of_assets, predicted_security = crypto_to_trade, prediction_type = "classification", long_only_classif_threshold = None, absolute_threshold = 0)

except_date = list(crypto_dataset_clean.columns)
except_date.remove("Date")
crypto_dataset_clean[except_date] = crypto_dataset_clean[except_date].apply(pd.to_numeric, errors='coerce')

model_instance = StrategySimulator(dataset = crypto_dataset_clean, 
                    model = MLPClassifier(), 
                    Y_name = crypto_to_trade + "_period_ahead_return", 
                    date_name = "Date",
                    bid_ask = 0.002,
                    trade_commission = 0.001)
filename = 'MLP_model_LT.pkl'
pickle.dump(model_instance.optimal_model, open(filename, 'wb'))

OOS_error, dummy_error = model_instance.time_series_tune(train_window = 600, test_window = 10, step = 10, OOS_window = 0.3, 
                                                        prediction_type = "classification", max_RS = 10, normalize = True, binary_rebalance = False)

#%%

asset_data_parent_ready = asset_data_parent.copy()
crypto_dataset_clean_reg = clean_data(crypto_dataset = crypto_dataset, non_crypto_dataset = asset_data_parent_ready, crypto_asset_list = ["BTC", "LTC", "ETH", "ZEC"], non_crypto_asset_list = list_of_assets, predicted_security = crypto_to_trade, prediction_type = "regression")
actual_returns_plugin = crypto_dataset_clean_reg[[crypto_to_trade + "_period_ahead_return", crypto_to_trade + "_Close", "Date"]]
actual_returns_plugin[crypto_to_trade + "_next_close"] = actual_returns_plugin[crypto_to_trade + "_Close"].shift(-1)
actual_returns_plugin = actual_returns_plugin[[crypto_to_trade + "_period_ahead_return", crypto_to_trade + "_Close", crypto_to_trade + "_next_close", "Date"]]
performance, dummy_performance = model_instance.classif_strat(train_window = 500, test_window = 10, step = 10, normalize = True, 
                                        actual_returns_plugin = actual_returns_plugin, binary_rebalance = False, direction = "long_only", optimize_threshold = False)

# %% Summary Stats

print("Model and Dummy OOS Prediction Performance: ", OOS_error, dummy_error)

print("---------------------------------------------------------")
print("|ML Model|")
# Model:
print("Average Return: ") 
print(performance.mean()[["actual", "strategy_returns"]])
print("Simple Sharpe: ")
print((performance.mean()/performance.std())[["actual", "strategy_returns"]])

# Dummy:
print("---------------------------------------------------------")
print("|Dummy Model|")
print("Average Return: ") 
print(dummy_performance.mean()[["actual", "strategy_returns"]])
print("Simple Sharpe: ")
print((dummy_performance.mean()/dummy_performance.std())[["actual", "strategy_returns"]])

# Model Graph Cumulative Returns:
cumulative_returns = performance.copy()
cumulative_returns[["actual", "strategy_returns"]] = cumulative_returns[["actual", "strategy_returns"]] + 1
cumulative_returns = cumulative_returns.sort_values(by = "Date").reset_index(drop = True)
cumulative_returns[["actual", "strategy_returns"]] = cumulative_returns[["actual", "strategy_returns"]].cumprod()


# Dummy Graph Cumulative Returns:
dummy_cumulative_returns = dummy_performance.copy()
dummy_cumulative_returns[["actual", "strategy_returns"]] = dummy_cumulative_returns[["actual", "strategy_returns"]] + 1
dummy_cumulative_returns = dummy_cumulative_returns.sort_values(by = "Date").reset_index(drop = True)
dummy_cumulative_returns[["actual", "strategy_returns"]] = dummy_cumulative_returns[["actual", "strategy_returns"]].cumprod()
dummy_cumulative_returns = dummy_cumulative_returns.rename(columns = {"strategy_returns":"dummy_returns"})

cumulative_returns = pd.merge(cumulative_returns[["actual", "Date", "strategy_returns"]], dummy_cumulative_returns[["Date", "dummy_returns"]], how = "inner", on = "Date") 
cumulative_returns = cumulative_returns[["Date", "strategy_returns", "dummy_returns","actual"]]

cumulative_returns[["strategy_returns", "dummy_returns","actual"]].plot()

print("Optimal model used: ", model_instance.optimal_model_text)

#%%

cumulative_returns.to_csv('outputs/backtest_plot.csv')
# model_load = pickle.load(open(filename, 'rb'))

# %% Regression Method

crypto_to_trade = "BTC"
predicted_field = "Close"
asset_data_parent_ready = asset_data_parent.copy()
crypto_dataset_clean = clean_data(crypto_dataset = crypto_dataset, non_crypto_dataset = asset_data_parent_ready, crypto_asset_list = ["BTC", "LTC", "ETH", "ZEC"], non_crypto_asset_list = list_of_assets, predicted_security = crypto_to_trade, 
                                predicted_field = predicted_field, prediction_type = "regression", long_only_classif_threshold = None, absolute_threshold = None)

except_date = list(crypto_dataset_clean.columns)
except_date.remove("Date")
crypto_dataset_clean[except_date] = crypto_dataset_clean[except_date].apply(pd.to_numeric, errors='coerce')

model_instance = StrategySimulator(dataset = crypto_dataset_clean, 
                    model = Lasso(), 
                    Y_name = crypto_to_trade + "_period_ahead_" + predicted_field, 
                    date_name = "Date",
                    bid_ask = 0.002,
                    trade_commission = 0.001)
filename = 'MLP_model_LT.pkl'
pickle.dump(model_instance.optimal_model, open(filename, 'wb'))

OOS_error, dummy_error = model_instance.time_series_tune(train_window = 200, test_window = 10, step = 10, OOS_window = 0.1, 
                                                        prediction_type = "regression", max_RS = 4, normalize = True, binary_rebalance = False)

print("Regression approach yielded a model error of " + float(OOS_error) + " and a naive error of " + float(dummy_error))

# %%
