"""
This code evaluates the results of the experiments based on several metrics.
"""
import copy
import shelve
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def acc_bias(link, dataset, valid_data, sens_attrs, metric="demographic_parity", lam=0.5, local_lam=0):
    groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
    actual_num_of_groups = len(groups)
    sensitive_groups = []
    sens_cols = groups.columns
    for i, row in groups.iterrows():
        sens_grp = []
        for col in sens_cols:
            sens_grp.append(row[col])
        sensitive_groups.append(tuple(sens_grp))

    if local_lam > 0:
        filename = link + "cluster.out"
        my_shelf = shelve.open(filename)
        for key in my_shelf:
            kmeans = my_shelf["kmeans"]
        my_shelf.close()

        clustered_df = dataset.groupby("cluster")

        cluster_count = 0
        total_size = 0
        cluster_ids = []
        for key, item in clustered_df:
            cluster_count += 1
            part_df = clustered_df.get_group(key)
            index_list = []
            for i, row in part_df.iterrows():
                index_list.append(i)
            df_local = valid_data.loc[index_list]
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
                                nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(knn_df.values)
                                indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                                real_indices = valid_data.index[indices].tolist()
                                for ind in real_indices[0]:
                                    index_list.append(ind)
            cluster_ids.append(index_list)

    loc = 1
    glob = 1
    error = 1
    if metric != "consistency":
        grouped_df = dataset.groupby(sens_attrs)
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
        for key, item in grouped_df:
            predsize = 0
            part_df = grouped_df.get_group(key)
            for i, row in part_df.iterrows():
                predsize += 1
                total_ppv = total_ppv + row["pred_label"]
                total_size = total_size + 1
                wrong_predicted = wrong_predicted + abs(row["pred_label"] - row["true_label"])
                if row["true_label"] == 0:
                    total_ppv_y0 = total_ppv_y0 + row["pred_label"]
                    total_size_y0 = total_size_y0 + 1
                    wrong_predicted_y0 = wrong_predicted_y0 + abs(row["pred_label"] - row["true_label"])
                    if row["pred_label"] == 1:
                        total_fp += 1
                        num_pos += 1
                    else:
                        num_neg += 1
                elif row["true_label"] == 1:
                    total_ppv_y1 = total_ppv_y1 + row["pred_label"]
                    total_size_y1 = total_size_y1 + 1
                    wrong_predicted_y1 = wrong_predicted_y1 + abs(row["pred_label"] - row["true_label"])
                    if row["pred_label"] == 0:
                        total_fn += 1
                        num_neg += 1
                    else:
                        num_pos += 1

        error = wrong_predicted/total_size

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
                model_ppv = model_ppv + row["pred_label"]
                model_size = model_size + 1
                if row["true_label"] == 0:
                    model_ppv_y0 = model_ppv_y0 + row["pred_label"]
                    model_size_y0 = model_size_y0 + 1
                    if row["pred_label"] == 1:
                        fp += 1
                elif row["true_label"] == 1:
                    model_ppv_y1 = model_ppv_y1 + row["pred_label"]
                    model_size_y1 = model_size_y1 + 1
                    if row["pred_label"] == 0:
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

        if metric == "demographic_parity":
            glob = dp/(len(grouped_df))
        elif metric == "equalized_odds":
            glob = eq_odd/(len(grouped_df))
        elif metric == "equal_opportunity":
            glob = eq_opp/(len(grouped_df))
        elif metric == "treatment_equality":
            glob = tr_eq/(len(grouped_df))

    elif metric == "consistency":
        consistency = 0
        for i, row_outer in dataset.iterrows():
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(valid_data.values)
            indices = nbrs.kneighbors(valid_data.loc[i].values.reshape(1, -1),\
                return_distance=False)
            real_indices = dataset.index[indices].tolist()
            df_local = dataset.loc[real_indices[0]]

            knn_ppv = 0
            knn_count = 0
            inacc = 0
            for j, row in df_local.iterrows():
                inacc += abs(row["pred_label"] - row["true_label"])
                knn_ppv = knn_ppv + row["pred_label"]
                knn_count = knn_count + 1
            knn_pppv = knn_ppv/knn_count
            consistency = consistency + abs(dataset.loc[i]["pred_label"] - knn_pppv)

        error = 0
        for i, row in dataset.iterrows():
            error += abs(row["pred_label"] - row["true_label"])
        error = error/len(dataset)
        glob = consistency/len(dataset)


    if local_lam > 0:
        loc = 0
        local_dp = 0
        local_eq_odd = 0
        local_eq_opp = 0
        local_tr_eq = 0
        for cluster in cluster_ids:
            cluster_df = dataset.loc[cluster]
            grouped_df = cluster_df.groupby(sens_attrs)
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
            for key, item in grouped_df:
                predsize = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    predsize += 1
                    total_ppv = total_ppv + row["pred_label"]
                    total_size = total_size + 1
                    wrong_predicted = wrong_predicted + abs(row["pred_label"] - row["true_label"])
                    if row["true_label"] == 0:
                        total_ppv_y0 = total_ppv_y0 + row["pred_label"]
                        total_size_y0 = total_size_y0 + 1
                        wrong_predicted_y0 = wrong_predicted_y0 + abs(row["pred_label"] - row["true_label"])
                        if row["pred_label"] == 1:
                            total_fp += 1
                            num_pos += 1
                        else:
                            num_neg += 1
                    elif row["true_label"] == 1:
                        total_ppv_y1 = total_ppv_y1 + row["pred_label"]
                        total_size_y1 = total_size_y1 + 1
                        wrong_predicted_y1 = wrong_predicted_y1 + abs(row["pred_label"] - row["true_label"])
                        if row["pred_label"] == 0:
                            total_fn += 1
                            num_neg += 1
                        else:
                            num_pos += 1

            error = wrong_predicted/total_size

            #Iterate again for formula
            count = 0
            dp = 0
            eq_odd = 0
            eq_opp = 0
            tr_eq = 0
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
                    model_ppv = model_ppv + row["pred_label"]
                    model_size = model_size + 1
                    if row["true_label"] == 0:
                        model_ppv_y0 = model_ppv_y0 + row["pred_label"]
                        model_size_y0 = model_size_y0 + 1
                        if row["pred_label"] == 1:
                            fp += 1
                    elif row["true_label"] == 1:
                        model_ppv_y1 = model_ppv_y1 + row["pred_label"]
                        model_size_y1 = model_size_y1 + 1
                        if row["pred_label"] == 0:
                            fn += 1

                dp = dp + abs(model_ppv/model_size - total_ppv/total_size)
                if model_size_y0 == 0:
                    eq_odd = (eq_odd + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
                elif model_size_y1 == 0:
                    eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0))
                else:
                    eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0)
                        + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
                if model_size_y1 == 0:
                    eq_opp = eq_opp + 0
                else:
                    eq_opp = eq_opp + abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1)
                if fp+fn == 0 and total_fp+total_fn == 0:
                    pass
                elif fp+fn == 0:
                    tr_eq = tr_eq + abs(0.5 - total_fp/(total_fp+total_fn))
                elif total_fp+total_fn == 0:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - 0.5)
                else:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))

            if metric == "demographic_parity":
                loc += dp/(len(grouped_df))
            elif metric == "equalized_odds":
                loc += eq_odd/(len(grouped_df))
            elif metric == "equal_opportunity":
                loc += eq_opp/(len(grouped_df))
            elif metric == "treatment_equality":
                loc += tr_eq/(len(grouped_df))

        loc = loc/len(cluster_ids)

    return 100 - (lam * error + ((1- lam) * (1 - local_lam)) * glob + (local_lam * (1 - lam)) * loc)*100
