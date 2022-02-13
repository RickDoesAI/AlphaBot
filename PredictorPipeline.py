#%% Imports

# General
import pandas as pd
import numpy as np
import datetime as dt
import random
from sklearn.metrics.pairwise import linear_kernel

# Statistical
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

#Supported models
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


#%% Time Series Plugin

class TimeCrossValidation():

    def __init__(self, input_data, train_window, 
                test_window, step = 1, date_column = "date"):
        
        self.input_data = input_data
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.date_column = date_column

    @staticmethod
    def frequency_guesser(input_data, date_column):
        temp_frequency_deriver = input_data.sort_values(by = date_column)
        temp_frequency_deriver["date_lag"] = temp_frequency_deriver[date_column].shift(1)
        temp_frequency_deriver["date_diff"] = (temp_frequency_deriver[date_column] - temp_frequency_deriver["date_lag"])/pd.Timedelta(hours=1)
        frequency = round(temp_frequency_deriver["date_diff"].mean())

        return frequency 

    def rolling_window_train_test_split(self):

        frequency = TimeCrossValidation.frequency_guesser(self.input_data, self.date_column)

        timespan = max(self.input_data[self.date_column]) - min(self.input_data[self.date_column])
        days, seconds = timespan.days, timespan.seconds
        timespan_hours = days*24 + seconds//3600
        
        super_index = []
        for i in range(0, timespan_hours - self.train_window*frequency - self.test_window*frequency - 1, self.step*frequency):
            adaptive_start = min(self.input_data[self.date_column]) + dt.timedelta(hours = i)
            train_endpoints = list([adaptive_start, adaptive_start + dt.timedelta(hours = self.train_window*frequency)])
            test_endpoints = list([adaptive_start + dt.timedelta(hours = self.train_window*frequency), 
                                    adaptive_start + dt.timedelta(hours = self.train_window*frequency + self.test_window*frequency)])

            train_test_tuple = (train_endpoints, test_endpoints)
            super_index.append(train_test_tuple)

        return super_index
    
    def expanding_window_train_test_split(self):

        pass

# splitter = TimeCrossValidation(input_data = model_ready_data, 
#                                 train_window = 5,
#                                 test_window = 5)
# train_test_split_index = splitter.rolling_window_train_test_split()

#%% Parent Predictor Class

class E2EModelDevelopment:

    def __init__(self, dataset, model, Y_name, date_name = None):

        self.dataset = dataset
        self.model = model
        self.Y_name = Y_name
        self.date_name = date_name
        
        self.optimal_model_text = None
        self.parameter_search_scores = None
        self.optimal_model = None
        self.dummy_model = None

    # 1. This section's functions are aimed at pre-model fit variable importance:

    def correlation_matrix(self):

        return (self.dataset.drop(self.date_name, axis = 1)).corr()

    def mutual_information_importance(self, n_best_features, nN, target_type = "continuous"):
        
        if len(self.Y_name) == 1 or type(self.Y_name) == str: # this method is only supported for non vector Y variables
            
            if self.date_name == None or self.date_name not in list(self.dataset.columns):
                variables_dataset = self.dataset.drop(self.Y_name, axis = 1)
                target_dataset = self.dataset[self.Y_name]
            else:
                variables_dataset = self.dataset.drop([self.Y_name, self.date_name], axis = 1)
                target_dataset = self.dataset[self.Y_name]

            if target_type == "continuous":
                mutual_information_matrix = mutual_info_regression(variables_dataset, target_dataset, n_neighbors= nN)
            elif target_type == "categorical":
                mutual_information_matrix = mutual_info_classif(variables_dataset, target_dataset, n_neighbors= nN)

            index_of_n_best_features = np.argpartition(mutual_information_matrix, -n_best_features)[-n_best_features:]
            most_important_features = variables_dataset.iloc[:,list(index_of_n_best_features)].columns

            final_importance_matrix = pd.concat([pd.DataFrame(variables_dataset.iloc[:,list(np.argpartition(mutual_information_matrix, -len(mutual_information_matrix))[-len(mutual_information_matrix):])].columns).reset_index(drop = True), 
                                                pd.DataFrame(mutual_information_matrix).reset_index(drop = True)], axis = 1)

        else:
            most_important_features = None
            final_importance_matrix = None
            print("This method is not supported for vector Y targets.")
            
        return most_important_features, final_importance_matrix # first output is to easily subset features for dimensionality reduction, second is for interpretational use

    def lasso_importance():

        pass #TODO

    # 2. This section's functions relate to prediction

    def train_val_test_split(self, test_split = 0.3):

        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

        self.X = self.dataset.drop(self.Y_name, axis = 1)
        self.Y = self.dataset[self.Y_name]

        dataset = pd.concat([self.X,self.Y], axis = 1)
        self.dataset_train_val = dataset.iloc[:round(len(dataset)*(1-test_split))]
        self.X_test = self.X.iloc[round(len(self.X)*(1-test_split)):]
        self.Y_test = self.Y.iloc[round(len(self.Y)*(1-test_split)):]

    def binary_class_rebalancing(self):

        majority_class = self.dataset_train_val[self.Y_name].mode()[0]
        minority_class_filtered_dataset = self.dataset_train_val[self.dataset_train_val[self.Y_name] != majority_class]
        majority_class_filtered_dataset = self.dataset_train_val[self.dataset_train_val[self.Y_name] == majority_class]
        majority_class_filtered_dataset = majority_class_filtered_dataset.sample(n = len(minority_class_filtered_dataset), replace = True)

        self.dataset_train_val = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis = 0)

    @staticmethod
    def parameter_grid_builder(model_object):

        #Random forest gridsearch parameters
        if "forest" in str(model_object).lower():
            param_grid = {
                'max_depth': [20,50, 100],
                'max_features': ["auto", "log2"],
                'n_estimators': [50, 100, 200]
            }

        #Neural network gridsearch parameters
        elif "mlp" in str(model_object).lower():
            param_grid = {
                'solver': ['adam','sgd'],
                'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                'alpha': [0.01, 0.1, 0.3, 0.5, 1, 5, 10],
                'hidden_layer_sizes': [(5,10,5,), (15,10,5,), (5,10,15), (5,10,30,10,5)]
            }

        elif "lasso" in str(model_object).lower():
            param_grid = {
                'alpha': [0, 0.5, 1, 10, 100, 500, 1000],
            }

        elif "logistic" in str(model_object).lower():
            param_grid = {
                'C': [0.01, 0.05, 0.1, 0.5, 1, 5,10,100],
            }            
        
        elif "neighbors" in str(model_object).lower():
            param_grid = {
                'n_neighbors': [int(1),int(2),int(3),int(5),int(10),int(20),int(30),int(50),int(100)],
            }            

        elif "gaussiannb" in str(model_object).lower():
            param_grid = {
                'var_smoothing': [1e-09, 0],
            }            

        return param_grid


    def cross_sectional_tune(self, k_fold_splits, normalize = False, reduce = False, reduced_size = None, decision_optimize = False, specificity_target = None):
        
        param_grid = E2EModelDevelopment.parameter_grid_builder(self.model)

        grid_search = GridSearchCV(estimator = self.model, param_grid = param_grid, 
                                cv = k_fold_splits, n_jobs = -1, verbose = 1)

        X_train_val = self.dataset_train_val.drop(self.Y_name, axis = 1)
        Y_train_val = self.dataset_train_val[self.Y_name]

        # METHOD: Dimensionality Reduction: sift the N most important features by mutual information score
        if reduce:

            def mutual_information(X, Y, n_best_features, nN):
                mutual_information_matrix = mutual_info_classif(X, Y, n_neighbors= nN)
                index_of_n_best_features = np.argpartition(mutual_information_matrix, -n_best_features)[-n_best_features:]
                return index_of_n_best_features

            feature_index = mutual_information(np.array(X_train_val), np.array(Y_train_val), reduced_size, 3)
            X_train_val = X_train_val.iloc[:,list(feature_index)]
            self.X_test = self.X_test.iloc[:,list(feature_index)]

        # METHOD: Normalization of the X feature set using a robust scaler 
        if normalize:

            transformer = RobustScaler().fit(X_train_val)

            X_train_val_columns = X_train_val.columns
            X_train_val = transformer.transform(X_train_val)
            X_train_val = pd.DataFrame(X_train_val)
            X_train_val.columns = X_train_val_columns

            X_test_columns = self.X_test.columns
            self.X_test = transformer.transform(self.X_test)
            self.X_test = pd.DataFrame(self.X_test)
            self.X_test.columns = X_test_columns

        grid_search.fit(X_train_val, Y_train_val)
        optimized_model = grid_search.best_estimator_
        optimized_model.fit(X_train_val, Y_train_val)

        # METHOD: Override to change the decision threshold of the model to favour type 1 or type 2 error
        if decision_optimize:

            candidate_threshold = 0
            specificity = 0
            while specificity < specificity_target:
                y_pred_proba = optimized_model.predict_proba(X_train_val)[:,1]
                y_pred = np.where(y_pred_proba < candidate_threshold, 0, 1)
                y_actual = Y_train_val.copy()

                tn, fp, fn, tp = confusion_matrix(y_pred = y_pred, y_true = y_actual).ravel()

                accuracy = (tp + tn)/(tp + tn + fp + fn)
                sensitivity = tp/(tp + fn)
                specificity = tn/(tn + fp)
                ppv = tp/(tp + fp)

                candidate_threshold = candidate_threshold + 0.01
            
            y_pred_proba = optimized_model.predict_proba(self.X_test)[:,1]
            y_pred = np.where(y_pred_proba < candidate_threshold, 0, 1)
            y_actual = self.Y_test.copy()

            tn, fp, fn, tp = confusion_matrix(y_pred = y_pred, y_true = y_actual).ravel()

            accuracy = (tp + tn)/(tp + tn + fp + fn)
            sensitivity = tp/(tp + fn)
            specificity = tn/(tn + fp)
            ppv = tp/(tp + fp)

        else:

            y_pred = optimized_model.predict(self.X_test)
            y_actual = self.Y_test.copy()

            tn, fp, fn, tp = confusion_matrix(y_pred = y_pred, y_true = y_actual).ravel()

            accuracy = (tp + tn)/(tp + tn + fp + fn)
            sensitivity = tp/(tp + fn)
            specificity = tn/(tn + fp)
            ppv = tp/(tp + fp)

        performance = pd.DataFrame([accuracy, sensitivity, specificity, ppv]).transpose()
        performance.columns = ["accuracy", "sensitivity", "specificity", "PPV"]

        metrics.plot_roc_curve(optimized_model, self.X_test, self.Y_test) 

        # METHOD: Permutation importance algorithm to extract salient features (just printed to console for ease)

        # print("Running Permutations...")
        # r = permutation_importance(optimized_model, X_train_val, Y_train_val, n_repeats=2, scoring = "accuracy")
        
        # r.importances_mean = (r.importances_mean/r.importances_mean.sum())*(accuracy - 0.5)

        # for i in r.importances_mean.argsort()[::-1]:
        #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        #         print(f"{self.X_test.columns[i]:<15}"
        #             f"{r.importances_mean[i]:.15f}"
        #             f" +/- {r.importances_std[i]:.15f}")


    def time_series_tune(self, train_window, test_window, step = 1, OOS_window = 0.3,
                        prediction_type = "classification", max_RS = 20, normalize = False, binary_rebalance = False):

        self.dataset = self.dataset.sort_values(by = self.date_name, ascending = True)
        self.IS_dataset = self.dataset.iloc[:round((1-OOS_window)*len(self.dataset)),:]
        self.OOS_dataset = self.dataset.iloc[round(((1-OOS_window)*len(self.dataset)) - (train_window + test_window)):,:]
        
        # Part 1: Search optimal parameter space

        splitter = TimeCrossValidation(input_data = self.IS_dataset, 
                                train_window = train_window,
                                test_window = test_window, step = step, date_column =  self.date_name)
        train_test_split_index = splitter.rolling_window_train_test_split()
        param_grid = E2EModelDevelopment.parameter_grid_builder(self.model)
        parameter_search_scores = pd.DataFrame()

        for i in range(0, max_RS):

            # RS parameter selection TODO: remove repetitions
            parameters = []
            for key in list(param_grid):
                parameters.append(random.choice(param_grid[key]))
            print(parameters)
            # Create parameter config
            parameter_config = pd.DataFrame()
            parameter_config = parameter_config.append(pd.DataFrame(parameters).transpose())
            parameter_config.columns = list(param_grid)
            cross_fold_eval_score_list = []
            cross_fold_y_pred = pd.DataFrame()
            cross_fold_y_actual = pd.DataFrame()
            for train_index, test_index in train_test_split_index:
                dataset_train = self.dataset[(self.dataset[self.date_name] >= train_index[0]) & (self.dataset[self.date_name] <= train_index[1])]

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

                # random search optimal parameters:

                # construct the model call with specified parameters:
                model_text = str(self.model).split(")")[0]
                for i in range(0, parameter_config.shape[1]):
                    if type(parameter_config.iloc[0,i]) == str:
                        model_text = model_text + str(parameter_config.columns[i]) + "=" + str("'" + str(parameter_config.iloc[0,i]) + "'" + ",")
                    else:
                        model_text = model_text + str(parameter_config.columns[i]) + "=" + str(str(parameter_config.iloc[0,i]) + ",")
                model_text = model_text[:-1] # removev final comma
                model_text = model_text + ")"

                if "neighbors" in model_text.lower():
                    opt_n_neighbors = parameter_config.iloc[0,0]
                    if prediction_type == "classification":
                        candidate_model = KNeighborsClassifier(n_neighbors = int(opt_n_neighbors))
                    else:
                        candidate_model = KNeighborsRegressor(n_neighbors = int(opt_n_neighbors))
                else:
                    candidate_model = eval(model_text)

                candidate_model.fit(X_train, Y_train)
                y_pred = candidate_model.predict(X_val)

                # if prediction_type == "classification":
                #     accuracy_dataframe = pd.DataFrame(y_pred == Y_val)
                #     eval_score = float(accuracy_dataframe.mean())
                # elif prediction_type == "regression":
                #     eval_score = median_absolute_error(Y_val, y_pred)

                # cross_fold_eval_score_list.append(eval_score)

                cross_fold_y_pred = cross_fold_y_pred.append(pd.DataFrame(y_pred))
                cross_fold_y_actual = cross_fold_y_actual.append(pd.DataFrame(Y_val))

            cross_fold_y_pred = cross_fold_y_pred.reset_index(drop = True)
            cross_fold_y_actual = cross_fold_y_actual.reset_index(drop = True)

            cross_fold_y_pred.columns = ["prediction"]
            if prediction_type == "classification":
                accuracy_dataframe = pd.DataFrame(cross_fold_y_pred.values == cross_fold_y_actual.values)
                final_eval_score = float(accuracy_dataframe.mean())
            elif prediction_type == "regression":
                final_eval_score = median_absolute_error(cross_fold_y_actual, cross_fold_y_pred)

            # final_eval_score = float(pd.DataFrame(cross_fold_eval_score_list).mean())

            parameter_score = parameters + [final_eval_score]
            appendable_score = pd.DataFrame(parameter_score).transpose()
            appendable_score.columns = list(param_grid) + ["eval_score"]
            parameter_search_scores = parameter_search_scores.append(appendable_score)
            parameter_search_scores.columns = list(param_grid) + ["eval_score"]

            self.parameter_search_scores = parameter_search_scores

        # Part 2: train optimal model

        splitter = TimeCrossValidation(input_data = self.OOS_dataset, 
                                train_window = train_window,
                                test_window = test_window, step = step, date_column =  self.date_name)
        train_test_split_index = splitter.rolling_window_train_test_split()

        if prediction_type == "classification":
            optimal_parameters = pd.DataFrame(parameter_search_scores.iloc[pd.to_numeric(parameter_search_scores["eval_score"]).idxmax(),:]).transpose()
            column_names = list(optimal_parameters.columns)
            column_names.remove("eval_score")
            optimal_parameters = optimal_parameters[column_names]
        elif prediction_type == "regression":
            optimal_parameters = pd.DataFrame(parameter_search_scores.iloc[pd.to_numeric(parameter_search_scores["eval_score"]).idxmin(),:]).transpose()
            column_names = list(optimal_parameters.columns)
            column_names.remove("eval_score")
            optimal_parameters = optimal_parameters[column_names]

        OOS_cross_fold_y_pred = pd.DataFrame()
        OOS_cross_fold_y_actual = pd.DataFrame()

        dummy_cross_fold_y_pred = pd.DataFrame()
        dummy_cross_fold_y_actual = pd.DataFrame()
        for train_index, test_index in train_test_split_index:
            dataset_train = self.dataset[(self.dataset[self.date_name] >= train_index[0]) & (self.dataset[self.date_name] <= train_index[1])]
            
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

                pre_normalized_X_val = X_val.copy()
                X_val_columns = X_val.columns
                X_val = transformer.transform(X_val)
                X_val = pd.DataFrame(X_val)
                X_val.columns = X_val_columns

            # construct the model call with optimal parameters:
            model_text = str(self.model).split(")")[0]
            for i in range(0, optimal_parameters.shape[1]):
                if type(optimal_parameters.iloc[0,i]) == str:
                    model_text = model_text + str(optimal_parameters.columns[i]) + "=" + str("'" + str(optimal_parameters.iloc[0,i]) + "'" + ",")
                else:
                    model_text = model_text + str(optimal_parameters.columns[i]) + "=" + str(str(optimal_parameters.iloc[0,i]) + ",")
            model_text = model_text[:-1] # removev final comma
            model_text = model_text + ")"
            self.optimal_model_text = model_text
            
            if "neighbors" in model_text.lower():
                    opt_n_neighbors = parameter_config.iloc[0,0]
                    if prediction_type == "classification":
                        candidate_model = KNeighborsClassifier(n_neighbors = int(opt_n_neighbors))
                    else:
                        candidate_model = KNeighborsRegressor(n_neighbors = int(opt_n_neighbors))
            else:
                candidate_model = eval(model_text)

            self.optimal_model = candidate_model
            candidate_model.fit(X_train, Y_train)
            y_pred = candidate_model.predict(X_val)

            OOS_cross_fold_y_pred = OOS_cross_fold_y_pred.append(pd.DataFrame(y_pred))
            OOS_cross_fold_y_actual = OOS_cross_fold_y_actual.append(pd.DataFrame(Y_val))

            # Dummy Model:
            if prediction_type == "classification":
                dummy_clf = DummyClassifier(strategy="most_frequent")
                self.dummy_model = dummy_clf
                dummy_clf.fit(X_train, Y_train)
                dummy_y_pred = dummy_clf.predict(X_val)
                dummy_cross_fold_y_pred = dummy_cross_fold_y_pred.append(pd.DataFrame(dummy_y_pred))
                dummy_cross_fold_y_actual = dummy_cross_fold_y_actual.append(pd.DataFrame(Y_val))
            if prediction_type == "regression":
                dummy_cross_fold_y_pred = dummy_cross_fold_y_pred.append(pd.DataFrame(pre_normalized_X_val["lag"]))
                dummy_cross_fold_y_actual = dummy_cross_fold_y_actual.append(pd.DataFrame(Y_val))

        OOS_cross_fold_y_pred = OOS_cross_fold_y_pred.reset_index(drop = True)
        OOS_cross_fold_y_actual = OOS_cross_fold_y_actual.reset_index(drop = True)

        dummy_cross_fold_y_pred = dummy_cross_fold_y_pred.reset_index(drop = True)
        dummy_cross_fold_y_actual = dummy_cross_fold_y_actual.reset_index(drop = True)

        OOS_cross_fold_y_pred.columns = ["prediction"]
        if prediction_type == "classification":
            accuracy_dataframe = pd.DataFrame(OOS_cross_fold_y_pred.values == OOS_cross_fold_y_actual.values)
            OOS_final_eval_score = float(accuracy_dataframe.mean())

            dummy_accuracy_dataframe = pd.DataFrame(dummy_cross_fold_y_pred.values == dummy_cross_fold_y_actual.values)
            dummy_final_eval_score = float(dummy_accuracy_dataframe.mean())
        elif prediction_type == "regression":
            OOS_final_eval_score = median_absolute_error(OOS_cross_fold_y_actual, OOS_cross_fold_y_pred)
            if "return" in self.Y_name.lower():
                dummy_final_eval_score = median_absolute_error(dummy_cross_fold_y_actual, dummy_cross_fold_y_pred)
            else:
                dummy_final_eval_score = median_absolute_error(dummy_cross_fold_y_actual, dummy_cross_fold_y_pred)

        return OOS_final_eval_score, dummy_final_eval_score


#%%