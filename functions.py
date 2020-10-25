# Import needed libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import sklearn.cluster
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Feature windows concatenation function
def augment_features_window(X,
                            N_neig):  # N_neig is window size - how about including it in gridsearch? Whaddaya reckon?

    # Parameters of row and column amount (where columns are features, so well-applicable to our case)
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding with window width-sized arrays from above and below
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat * (2 * N_neig + 1)))
    for r in np.arange(N_row) + N_neig:  # starting from the first non-padding row?
        this_row = []
        for c in np.arange(-N_neig, N_neig + 1):  # window of columns
            this_row = np.hstack((this_row, X[r + c]))  # add values in window to the row-vector
        X_aug[r - N_neig] = this_row  # remove padding safeguard

    return X_aug  # overall works like a CNN pooling layer of sorts, but concatenates instead of performing num. transforms?


# Feature gradient computation function
def augment_features_gradient(X, depth):
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))  # return the first difference of depths - essentially interval lengths
    d_diff[d_diff == 0] = 0.001  # secure against division by zero
    X_diff = np.diff(X, axis=0)  # compute differences of values
    X_grad = X_diff / d_diff  # approximate derivative by the best possible difference quotient

    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    X_diff2 = np.diff(X_grad, axis=0)
    X_grad2 = X_diff2 / d_diff

    X_grad2 = np.concatenate((X_grad2, np.zeros((1, X_grad2.shape[1]))))

    return X_grad, X_grad2


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1] * (N_neig * 2 + 2)))  # augment window
    for w in np.unique(well):  # loop over wells
        w_idx = np.where(well == w)[0]  # well index

        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])  # get gradient for the well
        X_aug[w_idx, :] = np.concatenate((X_aug_grad), axis=1)  # add a new column (along column axis)

    # Find padded rows (for later removal via setdiff1d)
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])

    return X_aug, padded_rows

def low_pass_filter_anomaly_detection(df,
                                      column_name,
                                      number_of_stdevs_away_from_mean):
    """
    Implement a low-pass filter to detect anomalies in a time series, and save the filter outputs
    (True/False) to a new column in the dataframe.
    Arguments:
        df: Pandas dataframe
        column_name: string. Name of the column that we want to detect anomalies in
        number_of_stdevs_away_from_mean: float. Number of standard deviations away from
        the mean that we want to flag anomalies at. For example, if
        number_of_stdevs_away_from_mean=2,
        then all data points more than 2 standard deviations away from the mean are flagged as
        anomalies.
    Outputs:
        df: Pandas dataframe. Dataframe containing column for low pass filter anomalies
        (True/False)
    """
    #60-day rolling average
    df[column_name+'_Rolling_Average']=df[column_name].rolling(window=60, center=True).mean()
    #60-day standard deviation
    df[column_name+'_Rolling_StDev']=df[column_name].rolling(window=60, center=True).std()
    #Detect anomalies by determining how far away from the mean (in terms of standard deviation)
    #each data point is
    df[column_name+'_Low_Pass_Filter_Anomaly']=(abs(df[column_name]-df[
                                column_name+'_Rolling_Average'])>(
                                number_of_stdevs_away_from_mean*df[
                                column_name+'_Rolling_StDev']))
    return df

def isolation_forest_anomaly_detection(df,
                                       column_name,
                                       outliers_fraction=0.05):
    """
    In this definition, time series anomalies are detected using an Isolation Forest algorithm.
    Arguments:
        df: Pandas dataframe
        column_name: string. Name of the column that we want to detect anomalies in
        outliers_fraction: float. Percentage of outliers allowed in the sequence.
    Outputs:
        df: Pandas dataframe with column for detected Isolation Forest anomalies (True/False)
    """
    #Scale the column that we want to flag for anomalies
    min_max_scaler = StandardScaler()
    np_scaled = min_max_scaler.fit_transform(df[[column_name]])
    scaled_time_series = pd.DataFrame(np_scaled)
    # train isolation forest
    model =  IsolationForest(contamination = outliers_fraction, behaviour='new')
    model.fit(scaled_time_series)
    #Generate column for Isolation Forest-detected anomalies
    isolation_forest_anomaly_column = column_name+'_Isolation_Forest_Anomaly'
    df[isolation_forest_anomaly_column] = model.predict(scaled_time_series)
    df[isolation_forest_anomaly_column] = df[isolation_forest_anomaly_column].map( {1: False, -1: True} )
    return df


def numeric_preproccessing(data):
    num_pipeline = Pipeline([
        (('scaler', RobustScaler()))
    ])

    ## cathegorical encoding
    le = LabelEncoder()
    formation = data['FORMATION'].tolist()
    formation = le.fit_transform(formation)
    group = data['GROUP'].tolist()
    group = le.fit_transform(group)
    ## deleting outliers

    # transform needed features
    formation = num_pipeline.fit_transform(formation.reshape(-1, 1))
    data['CALI'] = num_pipeline.fit_transform(data['CALI'].values.reshape(-1, 1))
    data['RMED'] = num_pipeline.fit_transform(data['RMED'].values.reshape(-1, 1))
    data['RDEP'] = num_pipeline.fit_transform(data['RDEP'].values.reshape(-1, 1))
    data['GR'] = num_pipeline.fit_transform(data['GR'].values.reshape(-1, 1))
    data['DEPTH_MD'] = num_pipeline.fit_transform(data['DEPTH_MD'].values.reshape(-1, 1))
    data['DTC'] = num_pipeline.fit_transform(data['DTC'].values.reshape(-1, 1))
    data['ROP'] = num_pipeline.fit_transform(data['ROP'].values.reshape(-1, 1))
    data['RHOB'] = num_pipeline.fit_transform(data['RHOB'].values.reshape(-1, 1))
    data['SP'] = num_pipeline.fit_transform(data['SP'].values.reshape(-1, 1))
    data['DRHO'] = num_pipeline.fit_transform(data['DRHO'].values.reshape(-1, 1))
    data['NPHI'] = num_pipeline.fit_transform(data['NPHI'].values.reshape(-1, 1))
    data['PEF'] = num_pipeline.fit_transform(data['PEF'].values.reshape(-1, 1))
    data['BS'] = num_pipeline.fit_transform(data['BS'].values.reshape(-1, 1))

    data['DTS'] = num_pipeline.fit_transform(data['DTS'].values.reshape(-1, 1))

    Z = data[['CALI', 'RMED', 'RDEP', 'GR', 'DTC', 'RHOB', 'DRHO', 'NPHI']]
    gr1, gr2 = augment_features_gradient(Z, data['DEPTH_MD'])

    # imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imput = imputer.fit(data[['X_LOC', 'Y_LOC']])
    data[['X_LOC', 'Y_LOC']] = imput.transform(data[['X_LOC', 'Y_LOC']])

    X = np.hstack((Z.values, gr1, gr2, data['DEPTH_MD'].values.reshape(-1,1)))

    return X


def Clusterisation(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imput = imputer.fit(data[['X_LOC', 'Y_LOC']])
    data[['X_LOC', 'Y_LOC']] = imput.transform(data[['X_LOC', 'Y_LOC']])
    X = data['CLUSTERS'] = kmcluster.predict(data[['X_LOC', 'Y_LOC']])
    return X