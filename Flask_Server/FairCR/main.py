"""
main method
"""
import warnings
import argparse
import ast
import copy
import itertools
import subprocess
import json
import shelve
import time
import joblib
import re
import random
import math
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from aif360.datasets import BinaryLabelDataset
from scipy.stats import pearsonr
import traceback

import algorithm
from algorithm.FALCC_files.parameter_estimation import log_means
from algorithm.evaluation.eval_classifier import acc_bias

from preprocess_main import preprocess

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-skriptsDirectory", type=str, help="Path to the Directory containg the scripts, datasets, etc")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--favored", type=str, help="Tuple of values of privileged group.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
parser.add_argument("--models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
parser.add_argument("--opt", default="False", type=str, help="Set to True if hyperparameter\
    tuning already has been performed on the dataset. Default: False")
parser.add_argument("--removal", default="False", type=str, help="Set to True if protected attributes\
    should be removed.")
parser.add_argument("--binarize", default="False", type=str, help="Set to True if protected attributes\
    should be binarized.")
parser.add_argument("--lam", default=0.5, type=float, help="Value of the fairness weight.")
parser.add_argument("--local_lam", default=0.0, type=float, help="Value of the local fairness weight.")
parser.add_argument("--balance", default="False", type=str, help="")
parser.add_argument("--fselect", default="False", type=str, help="")
parser.add_argument("--dimred", default="False", type=str, help="")
parser.add_argument("--cluster", default="KMeans", type=str, help="")
parser.add_argument("--ca", default="LOGmeans", type=str, help="")
parser.add_argument("--proxy", default="no", type=str, help="")
args = parser.parse_args()

input_file = args.ds
main_directory_path = args.skriptsDirectory
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
favored = ast.literal_eval(args.favored)
label = args.label
metric = args.metric
randomstate = args.randomstate
if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)
model_list = ast.literal_eval(args.models)
tuning = args.tuning == "True"
opt = args.opt == "True"
rem_prot = args.removal == "True"
binarize = args.binarize == "True"
lam = args.lam
local_lam = args.local_lam
balance = args.balance
fselect = args.fselect
dimred = args.dimred
cluster = args.cluster
ca = args.ca
proxy = args.proxy

model_dict = dict()

df = pd.read_csv(main_directory_path + "/Datasets" + "/" + input_file + ".csv", index_col=index)


error_df = pd.read_csv(main_directory_path + "/configs/ERRORS.csv")

#Also set based on meta features
setting = dict()
setting["randomstate"] = randomstate
setting["binarize"] = False
if balance == "False":
    setting["balance"] = False
else:
    setting["balance"] = True
    if "Classic" in balance:
        setting["balance_type"] = "classic"
    else:
        setting["balance_type"] = "adapted"
    if "Def" in balance:
        setting["balance_method"] = "Default"
    elif "ADAS" in balance:
        setting["balance_method"] = "ADASYN"
    elif "ENN" in balance:
        setting["balance_method"] = "ENN"
    elif "CC" in balance:
        setting["balance_method"] = "ClusterCentroids"
    
if fselect == "VT":
    method = "VarianceThreshold"
if fselect == "RFECV":
    method = "RFECV"
else:
    method = False
setting["feature_selection"] = method
if dimred == "True":
    method = True
else:
    method = False
setting["dim_reduction"] = method

#Perform some preprocessing of the data
(X_train2, y_train2,
    X_val2, y_val2,
    X_test2, y_test2,
    dataset_train, dataset_val, dataset_test,
    val_df, result_df,
    sens_attrs, favored,
    privileged_groups, unprivileged_groups) = preprocess(df, sens_attrs, favored, label, setting)


tune_eval_df = pd.DataFrame()
tune_eval_df["index"] = y_train2.index.tolist()
tune_eval_df = tune_eval_df.set_index("index")
for sens in sens_attrs:
    tune_eval_df[sens] = X_train2[sens]
tune_eval_df["true_label"] = y_train2[label]

#Shelve all variables for the validation data and save it the folder.
filename = link + "cluster.out"
my_shelf = shelve.open(filename)
if "kmeans" not in my_shelf.keys():
    sens_groups = len(X_val2.groupby(sens_attrs))
    if proxy == "reweigh":
        with open(link + "reweighing_attributes.txt", 'w') as outfile:
            df_new = copy.deepcopy(df)
            cols = list(df_new.columns)
            cols.remove(label)
            for sens in sens_attrs:
                cols.remove(sens)

            for col in cols:
                x_arr = df_new[col].to_numpy()
                col_diff = 0
                for sens in sens_attrs:
                    z_arr = df_new[sens]
                    sens_corr = abs(pearsonr(x_arr, z_arr)[0])
                    if math.isnan(sens_corr):
                        sens_corr = 1
                    col_diff += (1 - sens_corr)
                col_weight = col_diff/len(sens_attrs)
                df_new[col] *= col_weight
                #X_val_new[col] *= col_weight
                outfile.write(col + ": " + str(col_weight) + "\n")
        df_new.to_csv(main_directory_path + "/Datasets/reweigh/" + input_file + ".csv", index_label=index)
    elif proxy == "remove":
        with open(link + "removed_attributes.txt", 'w') as outfile:
            df_new = copy.deepcopy(df)
            cols = list(df_new.columns)
            cols.remove(label)
            for sens in sens_attrs:
                cols.remove(sens)

            for col in cols:
                cont = False
                x_arr = df_new[col].to_numpy()
                col_diff = 0
                for sens in sens_attrs:
                    z_arr = df_new[sens]
                    pearson = pearsonr(x_arr, z_arr)
                    sens_corr = abs(pearson[0])
                    if math.isnan(sens_corr):
                        sens_corr = 1
                    if sens_corr > 0.5 and pearson[1] < 0.05:
                        #X_val_new = X_val_new.loc[:, X_val_new.columns != col]
                        df_new = df_new.loc[:, df_new.columns != col]
                        cont = True
                        outfile.write(col + "\n")
                        continue
            df_new.to_csv(main_directory_path + "/Datasets/removed/" + input_file + ".csv", index_label=index)

    X_val_cluster = copy.deepcopy(X_val2)
    for sens in sens_attrs:
        X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != sens]
    
    min_cluster = min(len(X_val2.columns), int(len(X_val2)/(50*sens_groups)))
    max_cluster = min(int(len(X_val2.columns)**2/2), int(len(X_val2)/(10*sens_groups)))
    clustersize = log_means(X_val_cluster, min_cluster, max_cluster)

    #Save the number of generated clusters as metadata
    with open(link + "clustersize.txt", 'w') as outfile:
        outfile.write(str(clustersize))

    #Apply the k-means algorithm on the validation dataset
    kmeans = KMeans(clustersize).fit(X_val_cluster)
    cluster_results = kmeans.predict(X_val_cluster)
    X_val_cluster["cluster"] = cluster_results
    tune_eval_df["cluster"] = cluster_results
    my_shelf["kmeans"] = kmeans
else:
    sens_groups = len(X_val2.groupby(sens_attrs))
    X_val_cluster = copy.deepcopy(X_val2)
    for sens in sens_attrs:
        X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != sens]
    kmeans = my_shelf["kmeans"]
    cluster_results = kmeans.predict(X_val_cluster)
    X_val_cluster["cluster"] = cluster_results
    tune_eval_df["cluster"] = cluster_results

my_shelf.close()


for sens in sens_attrs:
    result_df[sens] = X_test2[sens]
df_dict = dict()
df_dict["filename"] = input_file
df_dict["sens_attrs"] = sens_attrs
df_dict["favored"] = favored
df_dict["label"] = label
df_dict["privileged_groups"] = privileged_groups
df_dict["unprivileged_groups"] = unprivileged_groups

log_regr = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
dectree = DecisionTreeClassifier()

params = json.load(open(main_directory_path + '/configs/params.json'))
opt_param = json.load(open(main_directory_path + '/configs/params_opt_' + metric + '.json'))
new_param = dict()

failed_df = pd.DataFrame()

try:
    time_df = pd.read_csv(link + "RUNTIME.csv")
    #mem_df = pd.read_csv(link + "MEMORY.csv")
    run_count = len(mem_df)
except:
    time_df = pd.DataFrame()
    #mem_df = pd.DataFrame()
    run_count = 0
model_to_clf = {}
for model in model_list:
    start = time.time()
    try:
        time_df.at[run_count, "dataset"] = input_file
        time_df.at[run_count, "model"] = model
        time_df.at[run_count, "metric"] = metric
        #mem_df.at[run_count, "dataset"] = input_file
        #mem_df.at[run_count, "model"] = model
        #mem_df.at[run_count, "metric"] = metric
        best_runtime = "ERROR"
        #mem_usage = 0
        sol = False
        #if model in del_models:
         #   continue
        print(model)
        if tuning:
            paramlist = list(params[model]["tuning"].keys())
            parameters = []
            if not opt:
                for param in paramlist:
                    parameters.append(params[model]["tuning"][param])
            else:
                for param in paramlist:
                    parameters.append([opt_param[model][input_file][str(randomstate)][param]])
            full_list = list(itertools.product(*parameters))
            do_eval = True
        else:
            paramlist = list(params[model]["default"].keys())
            li = []
            for param in paramlist:
                li.append(params[model]["default"][param])
            full_list = [li]
            do_eval = False

        real = [item for sublist in dataset_test.labels.tolist() for item in sublist]
        max_val = 0
        best_li = 0

        for i, li in enumerate(full_list):
            iteration_start = time.time()
            score = 0
            #func = eval(params[model]["method"])
            try:
                #AIF Preprocessing Predictions
                if model == "DisparateImpactRemover":
                    clf = algorithm.AIF_DisparateImpactRemover(df_dict, log_regr, repair=li[0], remove=rem_prot)
                elif model == "LFR":
                    clf = algorithm.AIF_LFR(df_dict, log_regr, k=li[3], Ax=li[0], Ay=li[1], Az=li[2], remove=rem_prot)
                elif model == "Reweighing":
                    clf = algorithm.AIF_Reweighing(df_dict, log_regr, remove=rem_prot)
                #AIF Inprocessing Predictions
                elif model == "AdversarialDebiasing":
                    clf = algorithm.AIF_AdversarialDebiasing(df_dict, "plain_classifier", adversary_loss=li[0], debias=li[1]=="True")
                elif model == "GerryFairClassifier":
                    clf = algorithm.AIF_GerryFairClassifier(df_dict, gamma=li[0])
                elif model == "MetaFairClassifier":
                    clf = algorithm.AIF_MetaFairClassifier(df_dict, metric, tau=li[0])
                elif model == "PrejudiceRemover":
                    clf = algorithm.AIF_PrejudiceRemover(df_dict, eta=li[0])
                elif model == "ExponentiatedGradientReduction":
                    clf = algorithm.AIF_ExponentiatedGradientReduction(df_dict, log_regr, metric, eps=li[0], learning_rate=li[1], remove=rem_prot)
                elif model == "GridSearchReduction":
                    clf = algorithm.AIF_GridSearchReduction(df_dict, log_regr, metric, lam=li[0], remove=rem_prot)
                #AIF Postprocessing Predictions
                elif model == "EqOddsPostprocessing":
                    clf = algorithm.AIF_EqOddsPostprocessing(df_dict, log_regr, remove=rem_prot, training=True)
                elif model == "CalibratedEqOddsPostprocessing":
                    clf = algorithm.AIF_CalibratedEqOddsPostprocessing(df_dict, log_regr, remove=rem_prot, training=True)
                elif model == "RejectOptionClassification":
                    clf = algorithm.AIF_RejectOptionClassification(df_dict, log_regr, metric, remove=rem_prot, training=True)
                #Other Preprocessing Predictions
                elif model == "Fair-SMOTE":
                    clf = algorithm.FairSMOTE(df_dict, log_regr, remove=rem_prot)
                elif model == "LTDD":
                    clf = algorithm.LTDD(df_dict, log_regr, remove=rem_prot)
                #Other Inprocessing predictions
                elif model == "FairGeneralizedLinearModel":
                    clf = algorithm.FairGeneralizedLinearModelClass(df_dict, lam=li[0], family=li[1], discretization=li[2])
                elif model == "FairnessConstraintModel":
                    clf = algorithm.FairnessConstraintModelClass(df_dict, c=li[0], tau=li[1], mu=li[2], eps=li[3])
                elif model == "DisparateMistreatmentModel":
                    clf = algorithm.DisparateMistreatmentModelClass(df_dict, c=li[0], tau=li[1], mu=li[2], eps=li[3])
                elif model == "FAGTB":
                    clf = algorithm.FAGTBClass(df_dict, estimators=li[0], learning_rate=li[1], lam=li[2], remove=rem_prot)
                elif model == "GradualCompatibility":
                    clf = algorithm.GradualCompatibility(df_dict, reg=li[0], reg_val=li[1], weights_init=li[2], lam=li[3])
                #Other Postprocessing predictions
                elif model == "JiangNachum":
                    clf = algorithm.JiangNachum(df_dict, log_regr, metric, estimators=li[0], learning_rate=li[1], remove=rem_prot)
                elif model == "FaX":
                    clf = algorithm.FaX(df_dict)
                X_train = copy.deepcopy(X_train2)
                X_test = copy.deepcopy(X_test2)
                y_train = copy.deepcopy(y_train2)
                y_test = copy.deepcopy(y_test2)
                X_val = copy.deepcopy(X_val2)
                y_val = copy.deepcopy(y_val2)

                #mem_usage = memory_usage((clf.fit, (X_train, y_train)), multiprocess=True, max_usage=True)
                clf.fit(X_train, y_train)
                val = clf.predict(X_val)
                pred = clf.predict(X_test)
                if model == "AdversarialDebiasing":
                    clf.close_sess()

                sol = True

                # store the clf in a dictionary
                clf_copy = copy.deepcopy(clf)
                model_to_clf[model] = clf_copy

            except Exception as e:
                print("------------------")
                pred = None
                failcount = len(failed_df)
                print("hier excepion")
                traceback.print_exc()
                failed_df.at[failcount, "model"] = model
                failed_df.at[failcount, "exceptions"] = e
                print(model)
                print(e)
                print(sklearn.__version__)
                print(pd.__version__)
                print("------------------")

            if tuning:
                tune_eval_df["pred_label"] = val
                score = acc_bias(link, tune_eval_df, X_train2, sens_attrs, metric, lam, local_lam)
                if score > max_val:
                    max_val = score
                    best_li = li
                    best_pred = pred
                    best_val = val
                    best_runtime = time.time() - iteration_start
                    #mem_df.at[run_count, "max_mem"] = mem_usage
                    if sol:
                        d_list = []
                        joblib_file = link + model + "_model_tuned.pkl"
                        joblib.dump(clf, joblib_file)
                        d_list.append(joblib_file)
                        d_list.append(val)
                        #Dictionary containing all models of the following form: {Model Name: [(1) Saved Model
                        #as .pkl, (2) Prediction of the model for our test data]
                        #Train and save each model on the training data set.
                        model_dict[joblib_file] = d_list

                if pred is not None:
                    result_df[model + "_" + str(i)] = pred
                    result_df.to_csv(link + model + "_" + str(i) + "_prediction.csv")
                    result_df = result_df.drop(columns=[model + "_" + str(i)])

            elif not tuning and sol:
                d_list = []
                joblib_file = link + model + "_model.pkl"
                joblib.dump(clf, joblib_file)
                d_list.append(joblib_file)
                d_list.append(val)
                #Dictionary containing all models of the following form: {Model Name: [(1) Saved Model
                #as .pkl, (2) Prediction of the model for our test data]
                #Train and save each model on the training data set.
                model_dict[joblib_file] = d_list

            if not tuning:
                best_runtime = time.time() - start
                #mem_df.at[run_count, "max_mem"] = mem_usage

        #Else no scoring returned
        if tuning and max_val > 0:
            result_df[model + "_tuned"] = best_pred
            result_df.to_csv(link + model + "_tuned_prediction.csv", index_label="index")
            result_df = result_df.drop(columns=[model + "_tuned"])
            new_param[model] = dict()
            for p, param in enumerate(paramlist):
                #opt_param[model][input_file][str(randomstate)][param] = best_li[p]
                new_param[model][param] = best_li[p]
            #with open('configs/params_opt_' + metric + '.json', 'w') as fp:
                #json.dump(opt_param, fp, indent=4)
            with open(link + 'params_opt.json', 'w') as fp:
                json.dump(new_param, fp, indent=4)
        elif pred is not None:
            result_df[model] = pred
            result_df.to_csv(link + model + "_prediction.csv", index_label="index")
            result_df = result_df.drop(columns=[model])

    except Exception as E:
        print(E)
        err_count = len(error_df)
        error_df.at[err_count, "dataset"] = input_file
        error_df.at[err_count, "model"] = model
        error_df.at[err_count, "error_type"] = str(type(E))
        error_df.at[err_count, "error_msg"] = str(E)

    time_df.at[run_count, "overall_time"] = time.time() - start
    time_df.at[run_count, "iteration_time"] = best_runtime
    run_count += 1

error_df.to_csv(main_directory_path + "/configs/ERRORS.csv", index=False)
#mem_df.to_csv(link + "MEMORY.csv", index=False)

X_test = copy.deepcopy(X_test2)
X_val = copy.deepcopy(X_val2)
y_val = copy.deepcopy(y_val2)
time_df.to_csv(link + "RUNTIME.csv", index=False)

# write some data in a shelve:
filename = link + "shelve.out"
my_shelf = shelve.open(filename)
my_shelf["x_test"] = X_test2
my_shelf["y_test"] = y_test2
my_shelf["kmeans"] = kmeans
# Also shelve every clf if possible
for model in model_to_clf:
    try:
        my_shelf[model] = model_to_clf[model]
    except Exception as e:
        print(f"Failed to shelve 'clf': {e}")
        my_shelf["model"] = None

my_shelf.close()
