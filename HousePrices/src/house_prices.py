#!/usr/bin/python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
import sklearn.feature_selection as fs
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from subprocess import check_output
from scipy.stats import norm

warnings.filterwarnings('ignore')
# Check output
#print(check_output(["ls", "../data"]).decode("utf8"))

FILE_DATA_TRAIN = "../data/train.csv"
FILE_DATA_TEST = "../data/test.csv"
OUTPUT_ATTRIBUTE = "SalePrice"
ID_ATTRIBUTE = "id"
OBJECT_TYPE = "object"

def get_object_feature(df, col, length):
    """
    Convert an object (categorical) feature into a int feature

    Parameters:
  
    Returns: 

    """
    # TODO: Verify name feature
    if df[col].dtype != 'object':
        print("The feature ", col, " is not an object feature.")
        return df
    elif len([i for i in df[col].T.notnull() if i == True]) != length:
        print("The feature ", col, " is missing data.")
        return df
    else:
        df_tmp = df
        counts = df_tmp[col].value_counts()
        df_tmp[col] = [counts.index.tolist().index(i) for i in df_tmp[col]]
        return df_tmp

def get_raw_data():
    """ 
    Get raw data from system
  
    Parameters: None
  
    Returns: 
  
    """
    data_train = pd.read_csv(FILE_DATA_TRAIN)
    data_test = pd.read_csv(FILE_DATA_TEST)
    return data_train, data_test

def remove_noise_and_drop_data(data_train):
    """
    Remove noises and drop from raw data
    There are some attribue having noise:
    + GrLivArea => data_train["GrLivArea"] > 4000
    + TotalBsmtSF => data_train["TotalBsmtSF"] > 3000
    + YearBuilt => data_train["YearBuilt"] < 1900

    Parameters:

    Returns: 

    """
    data_train.drop(data_train[data_train.GrLivArea > 4000].index, inplace = True)
    data_train.reset_index(drop = True, inplace = True)
    data_train.drop(data_train[data_train.TotalBsmtSF>3000].index, inplace = True)
    data_train.reset_index(drop = True, inplace = True)
    data_train.drop(data_train[data_train.YearBuilt < 1900].index, inplace = True)
    data_train.reset_index(drop = True, inplace = True)
    return data_train

def remove_missing_attribute_and_convert_categorical(data_train):
    included_features = [col for col in list(data_train)
                    if len([i for i in data_train[col].T.notnull() if i == True]) == len(data_train)
                        and col != OUTPUT_ATTRIBUTE and col != ID_ATTRIBUTE]
    data_processed = data_train[included_features]
    for col in list(data_processed):
        if data_processed[col].dtype == OBJECT_TYPE:
            data_processed = get_object_feature(data_processed, col, len(data_processed))

    return data_processed

def scale_ouput_by_log_transform(output):
    output = np.log(output)
    yt = [i for i in output]
    return yt

def get_attrubute_by_mutual_information(data_train, output):
    mir_result = fs.mutual_info_regression(data_train, output)
    feature_scores = []
    for i in np.arange(len(data_train.columns)):
        feature_scores.append([data_train.columns[i], mir_result[i]])
    sorted_scores = sorted(np.array(feature_scores), key = lambda s: float(s[1]), reverse = True)
    return sorted_scores

estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

def get_model(sorted_scores, data_train, num_feature):
    included_features = np.array(sorted_scores)[:, 0][:num_feature]
    X = data_train[included_features]
    Y = data_train['SalePrice']
    for col in list(X):
        if X[col].dtype == 'object':
            X = get_object_feature(X, col)

    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    mean_rfrs = []
    std_rfrs_upper = []
    std_rfrs_lower = []
    # Convert pandas.core.series.Series to list
    yt = [i for i in Y]
    np.random.seed(11111)
    for i in estimators:
        model = RandomForestRegressor(n_estimators = i, max_depth = None)
        scores_rfr = cross_val_score(model, X, yt, cv = 10, scoring = 'explained_variance')
        mean_rfrs.append(scores_rfr.mean())
        std_rfrs_lower.append(scores_rfr.mean() - scores_rfr.std()*2)
        std_rfrs_upper.append(scores_rfr.mean() + scores_rfr.std()*2)
    return mean_rfrs, std_rfrs_upper, std_rfrs_lower

def plot_accuracy(mean_rfrs, std_rfrs_upper, std_rfrs_lower, num_feature):
    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111)
    ax.plot(estimators, mean_rfrs, marker = 'o', linewidth = 4, markersize=12)
    ax.fill_between(estimators, std_rfrs_lower, std_rfrs_upper, facecolor = 'green', alpha = 0.3, interpolate = True)
    ax.set_ylim([-.2, 1])
    ax.set_xlim([0, 80])
    plt.title('Expected variance of Random Forest Regression: Top %d Feature'%num_feature)
    plt.ylabel('Expected Variance')
    plt.ylabel('Trees in Forest')
    plt.grid()
    plt.show()
    return

# mean_rfrs, std_rfrs_upper, std_rfrs_lower = get_model(sorted_scores, data_train, 20)
# plot_accuracy(mean_rfrs, std_rfrs_upper, std_rfrs_lower, 20)

# Test
# plt.scatter(data_train['TotalBsmtSF'], data_train['SalePrice'], c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("TotalBsmtSF")
# plt.ylabel("SalePrice")
# plt.show()

def main():
    data_train, data_test = get_raw_data()
    print(data_train.shape)
    data_train = remove_noise_and_drop_data(data_train)
    yt = scale_ouput_by_log_transform(data_train[OUTPUT_ATTRIBUTE])
    data_train = remove_missing_attribute_and_convert_categorical(data_train)
    sorted_scores = get_attrubute_by_mutual_information(data_train, yt)
    print(np.array(sorted_scores))

if __name__ == '__main__':
    main()
