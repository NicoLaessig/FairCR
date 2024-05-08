import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from preprocessing import GeneralPreprocessing
from aif360.datasets import BinaryLabelDataset

def preprocess(df, sens_attrs, favored, label, setting):
    """
    information [...]

    References:
        ...
    """
    #1. Impute missing data
    imputer = KNNImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    #1.5 Outlier_correction ???
    #TODO

    #Read the input dataset & split it into training, test & prediction dataset.
    #Currently testsize is hardcoded and train and val data is the same
    X = df.loc[:, df.columns != label]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
        random_state=setting["randomstate"])

    gproc = GeneralPreprocessing(X_train, y_train, sens_attrs, favored, label)

    grouped_df = df.groupby(sens_attrs)
    group_keys = grouped_df.groups.keys()

    #2. Binarize the label
    if setting["binarize"] and len(sens_attrs) > 1:
        X_train, sens_attrs, favored = gproc.binarize()

    #3. Standardize the dataset
    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    #4. Balance the dataset
    if setting["balance"]:
        X_train, y_train = gproc.balance(setting["balance_method"], setting["balance_type"])

    #5. Dimensionality reduction
    if setting["dim_reduction"]:
        X_train, X_test = gproc.dimensionality_reduction(X_test)

    #6. Feature selection
    if setting["feature_selection"]:
        X_train, X_test = gproc.feature_selection(setting["feature_selection"], X_test)

    #Convert to the right format
    X_val = copy.deepcopy(X_train)
    y_val = copy.deepcopy(y_train)
    train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)
    test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
    dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_val = BinaryLabelDataset(df=val_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
    full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)
    y_train = y_train.to_frame()
    y_val = y_val.to_frame()
    y_test = y_test.to_frame()
    result_df = copy.deepcopy(y_test)

    privileged_groups = []
    unprivileged_groups = []
    priv_dict = dict()
    unpriv_dict = dict()

    if isinstance(favored, tuple):
        for i, fav_val in enumerate(favored):
            priv_dict[sens_attrs[i]] = fav_val
            all_val = list(df.groupby(sens_attrs[i]).groups.keys())
            for poss_val in all_val:
                if poss_val != fav_val:
                    unpriv_dict[sens_attrs[i]] = poss_val
    else:
        if favored == 0:
            priv_dict[sens_attrs[0]] = 0
            unpriv_dict[sens_attrs[0]] = 1
        elif favored == 1:
            priv_dict[sens_attrs[0]] = 1
            unpriv_dict[sens_attrs[0]] = 0

    privileged_groups = [priv_dict]
    unprivileged_groups = [unpriv_dict]

    return (X_train, y_train, X_val, y_val, X_test, y_test, dataset_train, dataset_val, dataset_test, 
        val_df, result_df, sens_attrs, favored, privileged_groups, unprivileged_groups)

    