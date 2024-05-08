"""
This code evaluates the results of the experiments based on several metrics.
"""
import warnings
import argparse
import ast
import copy
import shelve
import pandas as pd
import time
import math
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-skriptsDirectory", type=str, help="Path to the Directory containg the scripts, datasets, etc")
parser.add_argument("--folder", type=str, help="Directory of the generated output files.")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
    Otherwise some metrics can't be used. Default: None.")
parser.add_argument("--models", default=None, type=str, help="List of models.")
parser.add_argument("--metric", type=str, help="Chosen fairness metric.")
parser.add_argument("--name", default="EVALUATION", type=str, help="Chosen evaluation file name.")
parser.add_argument("--proxy", default="no", type=str, help="Choose the proxy strategy for FALCC.")
args = parser.parse_args()

input_file = args.ds
main_directory_path = args.skriptsDirectory
link = args.folder
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
label = args.label
favored = ast.literal_eval(args.favored)
model_list = ast.literal_eval(args.models)
metric = args.metric
name = args.name
proxy = args.proxy
#Read the original dataset
if proxy == "no":
    original_data = pd.read_csv(main_directory_path + "/Datasets/" + input_file + ".csv", index_col=index)
elif proxy == "reweigh":
    original_data = pd.read_csv(main_directory_path + "/Datasets/reweigh/" + input_file + ".csv",  index_col=index)
elif proxy == "remove":
    original_data = pd.read_csv(main_directory_path + "/Datasets/removed/" + input_file + ".csv",  index_col=index)
dataset = copy.deepcopy(original_data)
original_data = original_data.drop(columns=[label])
for sens in sens_attrs:
    original_data = original_data.loc[:, original_data.columns != sens]

for i, model in enumerate(model_list):
    try:
        original_data_short = pd.read_csv(link + model + "_prediction.csv", index_col=index)
        break
    except:
        pass
original_data_short = pd.merge(original_data_short, original_data, left_index=True, right_index=True)
original_data_short = original_data_short.loc[:, original_data_short.columns != model_list[0]]
orig_datas = copy.deepcopy(original_data_short)
original_data_short = original_data_short.loc[:, original_data_short.columns != label]
valid_data = dataset.loc[orig_datas.index, :]

for sens in sens_attrs:
    original_data_short = original_data_short.loc[:, original_data_short.columns != sens]
dataset2 = copy.deepcopy(original_data_short)
total_size = len(original_data_short)

groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
actual_num_of_groups = len(groups)
sensitive_groups = []
sens_cols = groups.columns
for i, row in groups.iterrows():
    sens_grp = []
    for col in sens_cols:
        sens_grp.append(row[col])
    sensitive_groups.append(tuple(sens_grp))

filename = link + "cluster.out"
my_shelf = shelve.open(filename)
for key in my_shelf:
    kmeans = my_shelf["kmeans"]
my_shelf.close()

cluster_results = kmeans.predict(dataset2)
dataset2["cluster"] = cluster_results
clustered_df = dataset2.groupby("cluster")

cluster_count = 0
total_size = 0
cluster_ids = []
for key, item in clustered_df:
    cluster_count += 1
    part_df = clustered_df.get_group(key)
    index_list = []
    for i, row in part_df.iterrows():
        index_list.append(i)
    df_local = orig_datas.loc[index_list]
    groups2 = df_local[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
    num_of_groups = len(groups2)
    cluster_sensitive_groups = []
    for i, row in groups2.iterrows():
        sens_grp = []
        for col in sens_cols:
            sens_grp.append(row[col])
        cluster_sensitive_groups.append(tuple(sens_grp))

    #If a cluster does not contain samples of all groups, it will take the k nearest neighbors
    #(default value = 15) to test the model combinations
    if num_of_groups != actual_num_of_groups:
        cluster_center = kmeans.cluster_centers_[key]
        for sens_grp in sensitive_groups:
            if sens_grp not in cluster_sensitive_groups:
                if len(sens_attrs) == 1:
                    sens_grp = sens_grp[0]
                grouped_df = valid_data.groupby(sens_attrs)
                for key_inner, item_inner in grouped_df:
                    if key_inner == sens_grp:
                        knn_df = grouped_df.get_group(key_inner)
                        for sens_attr in sens_attrs:
                            knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                        knn_df = knn_df.loc[:, knn_df.columns != "index"]
                        knn_df = knn_df.loc[:, knn_df.columns != label]
                        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(knn_df.values)
                        indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                        real_indices = valid_data.index[indices].tolist()
                        for ind in real_indices[0]:
                            index_list.append(ind)
    cluster_ids.append(index_list)

df_count = 0
result_df = pd.DataFrame()
if metric != "consistency":
    for model in model_list:
        result_df.at[df_count, "model"] = model
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col=index)
        except:
            continue

        df = pd.merge(df, original_data_short, left_index=True, right_index=True)

        if actual_num_of_groups == 2:
            ds = StandardDataset(df, 
                label_name=label, 
                favorable_classes=[1], 
                protected_attribute_names=sens_attrs, 
                privileged_classes=[[favored]])

            dataset_pred = ds.copy()
            
            dataset_pred.labels = df[model]

            attr = dataset_pred.protected_attribute_names[0]
            idx = dataset_pred.protected_attribute_names.index(attr)
            privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
            unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

            metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            class_metric_pred = ClassificationMetric(ds, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

            result_df.at[df_count, "error_rate"] = class_metric_pred.error_rate() * 100
            result_df.at[df_count, "demographic_parity"] = abs(metric_pred.statistical_parity_difference()) * 100
            result_df.at[df_count, "equalized_odds"] = abs(class_metric_pred.average_abs_odds_difference()) * 100
            result_df.at[df_count, "equal_opportunity"] = abs(class_metric_pred.equal_opportunity_difference()) * 100
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            result_df.at[df_count, "treatment_equality"] = abs(treq1 - treq2) * 100
            fp = class_metric_pred.num_false_positives()
            fn = class_metric_pred.num_false_negatives()
            result_df.at[df_count, "tp"] = class_metric_pred.num_true_positives()
            result_df.at[df_count, "tn"] = class_metric_pred.num_true_negatives()
            result_df.at[df_count, "fp"] = fp
            result_df.at[df_count, "fn"] = fn
            result_df.at[df_count, "FPFN_balance_loss"] = abs(0.5 - fp/(fp+fn)) * 100

            result_df.at[df_count, "generalized_entropy_index"] = abs(class_metric_pred.generalized_entropy_index()) * 100
            result_df.at[df_count, "smoothed_edf"] = abs(metric_pred.smoothed_empirical_differential_fairness()) * 100
            
            result_df.at[df_count, "num_positives"] = abs(metric_pred.num_positives())
            result_df.at[df_count, "num_negatives"] = abs(metric_pred.num_negatives())
            
            result_df.at[df_count, "false_discovery_rate"] = abs(class_metric_pred.false_discovery_rate()) * 100
            result_df.at[df_count, "false_omission_rate"] = abs(class_metric_pred.false_omission_rate()) * 100
            result_df.at[df_count, "false_discovery_rate_difference"] = abs(class_metric_pred.false_discovery_rate_difference()) * 100
            result_df.at[df_count, "false_omission_rate_difference"] = abs(class_metric_pred.false_omission_rate_difference()) * 100

            #result_df.at[df_count, "average_predictive_value_difference"] = abs(class_metric_pred.average_predictive_value_difference()) * 100
            result_df.at[df_count, "between_all_groups_coefficient_of_variation"] = abs(class_metric_pred.between_all_groups_coefficient_of_variation()) * 100
            result_df.at[df_count, "between_all_groups_generalized_entropy_index"] = abs(class_metric_pred.between_all_groups_generalized_entropy_index()) * 100
            result_df.at[df_count, "between_all_groups_theil_index"] = abs(class_metric_pred.between_all_groups_theil_index()) * 100
            result_df.at[df_count, "between_group_coefficient_of_variation"] = abs(class_metric_pred.between_group_coefficient_of_variation()) * 100
            result_df.at[df_count, "between_group_generalized_entropy_index"] = abs(class_metric_pred.between_group_generalized_entropy_index()) * 100
            result_df.at[df_count, "differential_fairness_bias_amplification"] = abs(class_metric_pred.differential_fairness_bias_amplification()) * 100
            result_df.at[df_count, "false_positive_rate_difference"] = class_metric_pred.false_positive_rate_difference() * 100
            result_df.at[df_count, "false_negative_rate_difference"] = class_metric_pred.false_negative_rate_difference() * 100

            lrd_dp = 0
            lrd_eod = 0
            lrd_eop = 0
            lrd_te = 0
            lrd_di = 0
            cc = 0
            cc2 = 0
            for i, clusters in enumerate(cluster_ids):
                cluster_df = df.loc[clusters]
                cluster_size = len(cluster_df)
                cc += cluster_size
                cc2 += cluster_size

                ds = StandardDataset(cluster_df, 
                    label_name=label, 
                    favorable_classes=[1], 
                    protected_attribute_names=sens_attrs, 
                    privileged_classes=[[favored]])

                dataset_pred = ds.copy()
                dataset_pred.labels = cluster_df[model]
                #dataset_pred.label = df[model]

                attr = dataset_pred.protected_attribute_names[0]
                idx = dataset_pred.protected_attribute_names.index(attr)
                privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
                unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

                metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                class_metric_pred = ClassificationMetric(ds, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

                lrd_dp += abs(metric_pred.statistical_parity_difference()) * 100 * cluster_size
                if not math.isnan(class_metric_pred.average_abs_odds_difference()):
                    lrd_eod += abs(class_metric_pred.average_abs_odds_difference()) * 100 * cluster_size
                    #lrd_eop += abs(class_metric_pred.equal_opportunity_difference()) * 100 * cluster_size
                    treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
                    treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
                    if np.isnan(treq1) and np.isnan(treq2):
                        lrd_te += 0
                    elif np.isnan(treq1):
                        lrd_te += abs(0.5 - treq2) * 100 * cluster_size
                    elif np.isnan(treq2):
                        lrd_te += abs(treq1 - 0.5) * 100 * cluster_size
                    else:
                        lrd_te += abs(treq1 - treq2) * 100 * cluster_size
                else:
                    cc2 -= cluster_size
                disparate_impact = abs(metric_pred.disparate_impact())
                if disparate_impact > 1:
                    disparate_impact = disparate_impact ** -1
                lrd_di += disparate_impact * 100 * cluster_size
                if not math.isnan(class_metric_pred.equal_opportunity_difference()):
                    lrd_eop += abs(class_metric_pred.equal_opportunity_difference()) * 100 * cluster_size

            result_df.at[df_count, "lrd_dp"] = lrd_dp / cc
            result_df.at[df_count, "lrd_eod"] = lrd_eod / cc2
            result_df.at[df_count, "lrd_eop"] = lrd_eop / cc
            result_df.at[df_count, "lrd_te"] = lrd_te / cc2
            result_df.at[df_count, "lrd_di"] = lrd_di / cc
        else:
            """
            TODO LRD einbauen von FALCC Paper.
            """
            result_df.at[df_count, "model"] = model
            if sens_attrs[0] not in df.columns:
                for sens in sens_attrs:
                    df[sens] = orig_datas[sens]
            grouped_df = df.groupby(sens_attrs)
            total_ppv = 0
            total_size = 0
            total_ppv_y0 = 0
            total_size_y0 = 0
            total_ppv_y1 = 0
            total_size_y1 = 0
            discr_ppv_y0 = 0
            discr_size_y0 = 0
            discr_ppv_y1 = 0
            discr_size_y1 = 0
            wrong_predicted = 0
            wrong_predicted_y0 = 0
            wrong_predicted_y1 = 0
            total_fp = 0
            total_fn = 0
            num_pos = 0
            num_neg = 0
            #counterfactual = 0
            group_predsize = []
            #Get the favored group to test against and also the averages over the whole dataset
            di = dict()
            for key, item in grouped_df:
                predsize = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    predsize += 1
                    total_ppv = total_ppv + row[model]
                    total_size = total_size + 1
                    wrong_predicted = wrong_predicted + abs(row[model] - row[label])
                    if row[label] == 0:
                        total_ppv_y0 = total_ppv_y0 + row[model]
                        total_size_y0 = total_size_y0 + 1
                        wrong_predicted_y0 = wrong_predicted_y0 + abs(row[model] - row[label])
                        if row[model] == 1:
                            total_fp += 1
                            num_pos += 1
                        else:
                            num_neg += 1
                    elif row[label] == 1:
                        total_ppv_y1 = total_ppv_y1 + row[model]
                        total_size_y1 = total_size_y1 + 1
                        wrong_predicted_y1 = wrong_predicted_y1 + abs(row[model] - row[label])
                        if row[model] == 0:
                            total_fn += 1
                            num_neg += 1
                        else:
                            num_pos += 1

            result_df.at[df_count, "error_rate"] = wrong_predicted/total_size * 100
            #Iterate again for formula
            count = 0
            dp = 0
            eq_odd = 0
            eq_opp = 0
            tr_eq = 0
            impact = 0
            fp = 0
            fn = 0
            for key, item in grouped_df:
                model_ppv = 0
                model_size = 0
                model_ppv_y0 = 0
                model_size_y0 = 0
                model_ppv_y1 = 0
                model_size_y1 = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    model_ppv = model_ppv + row[model]
                    model_size = model_size + 1
                    if row[label] == 0:
                        model_ppv_y0 = model_ppv_y0 + row[model]
                        model_size_y0 = model_size_y0 + 1
                        if row[model] == 1:
                            fp += 1
                    elif row[label] == 1:
                        model_ppv_y1 = model_ppv_y1 + row[model]
                        model_size_y1 = model_size_y1 + 1
                        if row[model] == 0:
                            fn += 1

                dp = dp + abs(model_ppv/model_size - total_ppv/total_size)
                eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0)
                    + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
                eq_opp = eq_opp + abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1)
                if fp+fn == 0 and total_fp+total_fn == 0:
                    pass
                elif fp+fn == 0:
                    tr_eq = tr_eq + abs(0.5 - total_fp/(total_fp+total_fn))
                elif total_fp+total_fn == 0:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - 0.5)
                else:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))

            result_df.at[df_count, "demographic_parity"] = dp/(len(grouped_df)) * 100
            result_df.at[df_count, "equalized_odds"] = eq_odd/(len(grouped_df)) * 100
            result_df.at[df_count, "equal_opportunity"] = eq_opp/(len(grouped_df)) * 100
            result_df.at[df_count, "treatment_equality"] = tr_eq/(len(grouped_df)) * 100

        df_count += 1

elif metric == "consistency":
    model_list2 = []
    for model in model_list:
        try:
            pd.read_csv(link + model + "_prediction.csv", index_col="index")
            model_list2.append(model)
        except:
            continue

    df = pd.read_csv(link + model_list2[0] + "_prediction.csv", index_col="index")
    for i, model in enumerate(model_list2):
        if i == 0:
            continue
        df2 = pd.read_csv(link + model + "_prediction.csv", index_col="index")
        df = pd.merge(df, df2[[model]], how="inner", left_index=True, right_index=True)

    #Now evaluate each model according to the metrics implemented.
    model_count = 0
    models_consistency = [0 for model in model_list2]
    #CONSISTENCY TEST, COMPARE PREDICTION TO PREDICTIONS OF NEIGHBORS
    consistency = 0
    for i, row_outer in df.iterrows():
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset2.values)
        indices = nbrs.kneighbors(dataset2.loc[i].values.reshape(1, -1),\
            return_distance=False)
        real_indices = df.index[indices].tolist()
        df_local = df.loc[real_indices[0]]
        model_count = 0
        for model in model_list2:
            knn_ppv = 0
            knn_count = 0
            inacc = 0
            for j, row in df_local.iterrows():
                inacc += abs(row[model] - row[label])
                knn_ppv = knn_ppv + row[model]
                knn_count = knn_count + 1
            knn_pppv = knn_ppv/knn_count
            models_consistency[model_count] = models_consistency[model_count] + abs(df.loc[i][model] - knn_pppv)
            model_count = model_count + 1

    model_count = 0
    for model in model_list2:
        result_df.at[model_count, "model"] = model
        error = 0
        for i, row in df.iterrows():
            error += abs(row[model] - row[label])
        result_df.at[model_count, "error_rate"] = error/len(df) * 100
        result_df.at[model_count, "consistency"] = models_consistency[model_count]/len(df) * 100
        model_count = model_count + 1

runtime_df = pd.read_csv(link + "RUNTIME.csv", index_col="model")
for i, row in result_df.iterrows():
    model = re.sub("_tuned", "", result_df.at[i, "model"])
    result_df.at[i, "model"] = model
    result_df.at[i, "runtime"] = runtime_df.at[model, "overall_time"]
result_df.to_csv(link + name + "_" + str(input_file) + ".csv")
