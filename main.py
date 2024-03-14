"""
main method
"""
import warnings
import argparse
import ast
import copy
import itertools
import datetime
import subprocess
import json
import shelve
import time
import joblib
import re
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from aif360.datasets import BinaryLabelDataset
import algorithm
from algorithm.FALCC_files.parameter_estimation import log_means
from algorithm.evaluation.eval_classifier import acc_bias

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
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
    parser.add_argument("--models", default=None, type=str, help="List of models that should be trained.")
    parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
    parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
        tuning should be performed. Else, default parameter values are used. Default: False")
    parser.add_argument("--opt", default="False", type=str, help="Set to True if hyperparameter\
        tuning already has been performed on the dataset. Default: False")
    parser.add_argument("--lam", default=0.5, type=float, help="Value of the fairness weight.")
    parser.add_argument("--local_lam", default=0.0, type=float, help="Value of the local fairness weight.")
    args = parser.parse_args()

    input_file = args.ds
    link = args.output
    testsize = float(args.testsize)
    index = args.index
    sens_attrs = ast.literal_eval(args.sensitive)
    favored = ast.literal_eval(args.favored)
    label = args.label
    metric = args.metric
    randomstate = args.randomstate
    model_list = ast.literal_eval(args.models)
    tuning = args.tuning == "True"
    opt = args.opt == "True"
    lam = args.lam
    local_lam = args.local_lam

    model_dict = dict()

    df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
    grouped_df = df.groupby(sens_attrs)
    group_keys = grouped_df.groups.keys()

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

    #Read the input dataset & split it into training, test & prediction dataset.
    #Prediction dataset only needed for evaluation, otherwise size is automatically 0.
    X = df.loc[:, df.columns != label]
    y = df[label]

    #Currently testsize is hardcoded
    #X_train2, X_testpred, y_train, y_testpred = train_test_split(X, y, test_size=0.5,
    #    random_state=randomstate)
    #X_val2, X_test2, y_val, y_test = train_test_split(X_testpred, y_testpred,
    #    test_size=0.3, random_state=randomstate)
    X_train2, X_test2, y_train, y_test = train_test_split(X, y, test_size=0.3,
        random_state=randomstate)
    X_val2 = copy.deepcopy(X_train2)
    y_val = copy.deepcopy(y_train)

    train_df = pd.merge(X_train2, y_train, left_index=True, right_index=True)
    val_df = pd.merge(X_val2, y_val, left_index=True, right_index=True)
    test_df = pd.merge(X_test2, y_test, left_index=True, right_index=True)
    dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_val = BinaryLabelDataset(df=val_df, label_names=[label], protected_attribute_names=sens_attrs)
    dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
    full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)

    y_train2 = y_train.to_frame()
    y_val2 = y_val.to_frame()
    y_test2 = y_test.to_frame()
    result_df = copy.deepcopy(y_test2)

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

    params = json.load(open('configs/params.json'))
    opt_param = json.load(open('configs/params_opt_' + metric + '.json'))

    failed_df = pd.DataFrame()

    try:
        time_df = pd.read_csv(link + "RUNTIME.csv")
        run_count = len(time_df)
    except:
        time_df = pd.DataFrame()
        run_count = 0

    model_memory_usages = []
    start = time.time()

    for model in model_list:
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
                    parameters.append([opt_param[model][input_file][param]])
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
                    clf = algorithm.AIF_DisparateImpactRemover(df_dict, log_regr, repair=li[0], remove=False)
                elif model == "LFR":
                    clf = algorithm.AIF_LFR(df_dict, log_regr, k=li[3], Ax=li[0], Ay=li[1], Az=li[2], remove=False)
                elif model == "Reweighing":
                    clf = algorithm.AIF_Reweighing(df_dict, log_regr, remove=False)
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
                    clf = algorithm.AIF_ExponentiatedGradientReduction(df_dict, log_regr, metric, eps=li[0], learning_rate=li[1], remove=False)
                elif model == "GridSearchReduction":
                    clf = algorithm.AIF_GridSearchReduction(df_dict, log_regr, metric, lam=li[0], remove=False)
                #AIF Postprocessing Predictions
                elif model == "EqOddsPostprocessing":
                    clf = algorithm.AIF_EqOddsPostprocessing(df_dict, log_regr, remove=False, training=True)
                elif model == "CalibratedEqOddsPostprocessing":
                    clf = algorithm.AIF_CalibratedEqOddsPostprocessing(df_dict, log_regr, remove=False, training=True)
                elif model == "RejectOptionClassification":
                    clf = algorithm.AIF_RejectOptionClassification(df_dict, log_regr, metric, remove=False, training=True)
                #Other Preprocessing Predictions
                elif model == "Fair-SMOTE":
                    clf = algorithm.FairSMOTE(df_dict, log_regr, remove=False)
                elif model == "LTDD":
                    clf = algorithm.LTDD(df_dict, log_regr, remove=False)
                #Other Inprocessing predictions
                elif model == "FairGeneralizedLinearModel":
                    clf = algorithm.FairGeneralizedLinearModelClass(df_dict, lam=li[0], family=li[1], discretization=li[2])
                elif model == "FairnessConstraintModel":
                    clf = algorithm.FairnessConstraintModelClass(df_dict, c=li[0], tau=li[1], mu=li[2], eps=li[3])
                elif model == "DisparateMistreatmentModel":
                    clf = algorithm.DisparateMistreatmentModelClass(df_dict, c=li[0], tau=li[1], mu=li[2], eps=li[3])
                elif model == "FAGTB":
                    clf = algorithm.FAGTBClass(df_dict, estimators=li[0], learning_rate=li[1], lam=li[2], remove=False)
                elif model == "GradualCompatibility":
                    clf = algorithm.GradualCompatibility(df_dict, reg=li[0], reg_val=li[1], weights_init=li[2], lam=li[3])
                #Other Postprocessing predictions
                elif model == "JiangNachum":
                    clf = algorithm.JiangNachum(df_dict, log_regr, metric, estimators=li[0], learning_rate=li[1], remove=False)
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
                #model_memory_usages.append({ "timestamp": datetime.datetime.now(), "dataset": input_file, "model": model, "metric": metric, "memory": mem_usage})
                sol = True
    

            except Exception as e:
                print("------------------")
                pred = None
                failcount = len(failed_df)
                failed_df.at[failcount, "model"] = model
                failed_df.at[failcount, "exceptions"] = e
                print(model)
                print(e)
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

                if pred is not None:
                    result_df[model + "_" + str(i)] = pred
                    result_df.to_csv(link + model + "_" + str(i) + "_prediction.csv")
                    result_df = result_df.drop(columns=[model + "_" + str(i)])

            if not tuning:
                best_runtime = time.time() - start
                #mem_df.at[run_count, "max_mem"] = mem_usage

        #Else no scoring returned
        if tuning and max_val > 0:
            result_df[model + "_tuned"] = best_pred
            result_df.to_csv(link + model + "_tuned_prediction.csv")
            result_df = result_df.drop(columns=[model + "_tuned"])
            if input_file not in opt_param[model]:
                opt_param[model][input_file] = dict()
            for p, param in enumerate(paramlist):
                opt_param[model][input_file][param] = best_li[p]
            with open('configs/params_opt_' + metric + '.json', 'w') as fp:
                json.dump(opt_param, fp, indent=4)
        elif pred is not None:
            result_df[model] = pred
            result_df.to_csv(link + model + "_prediction.csv")
            result_df = result_df.drop(columns=[model])

        time_df.at[run_count, "overall_time"] = time.time() - start
        time_df.at[run_count, "iteration_time"] = best_runtime
        run_count += 1


    time_df.to_csv(link + "RUNTIME.csv", index=False)
    #mem_df.to_csv(link + "MEMORY.csv", index=False)


    #Shelve all variables and save it the folder.
    filename = link + "cluster.out"
    my_shelf = shelve.open(filename)
    if "kmeans" not in my_shelf.keys():
        sens_groups = len(X_test.groupby(sens_attrs))
        X_test_cluster = copy.deepcopy(X_test)
        for sens in sens_attrs:
            X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != sens]
        min_cluster = min(len(X_test.columns), int(len(X_test)/(50*sens_groups)))
        max_cluster = min(int(len(X_test.columns)**2/2), int(len(X_test)/(10*sens_groups)))
        clustersize = log_means(X_test_cluster, min_cluster, max_cluster)

        #Save the number of generated clusters as metadata
        with open(link + "clustersize.txt", 'w') as outfile:
            outfile.write(str(clustersize))

        #Apply the k-means algorithm on the validation dataset
        kmeans = KMeans(clustersize).fit(X_test_cluster)
        cluster_results = kmeans.predict(X_test_cluster)
        X_test_cluster["cluster"] = cluster_results

        my_shelf["kmeans"] = kmeans
    my_shelf.close()
