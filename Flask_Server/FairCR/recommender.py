"""
main method
"""
import copy
import psutil
import numpy as np
import pandas as pd

def generate_general_recommendation_plots(
    metrics, tuning, lam, local, datasize, mem, time, general_results="Summary/General_Results.csv",
    general_mem_results="Summary/General_Memory_Results.csv", general_time_results="Summary/General_Runtime_Results.csv"):
    """
    Generate general recommendation plots for fairness algorithms based on chosen parameters.

    Parameters:
    -----------
    models : list
        List of fairness algorithm models to include in the plots.
    metrics : list
        List of metrics to use for ranking the fairness algorithms.
    tuning : bool
        Whether or not hyperparameters were tuned.
    lam : float
        Weight for balancing global and local fairness.
    local : float
        Weight for balancing local fairness and error rate.
    datasize : int
        Size of the dataset in samples/rows.
    mem : float
        Maximum allowed memory usage as a fraction of total virtual memory.
    time : float
        Weight for balancing expected runtime and score.
    #We leave the following 3 at the default settings for the demo?
    general_results : str, optional
        Path to the CSV file containing general results, by default "Summary/General_Results.csv".
    general_mem_results : str, optional
        Path to the CSV file containing general memory results, by default "Summary/General_Memory_Results.csv".
    general_time_results : str, optional
        Path to the CSV file containing general runtime results, by default "Summary/General_Runtime_Results.csv".

    Returns:
    --------
    list
        A list of plotly bar plots and dataframes, one for each metric.
    """
    full_df = pd.read_csv(general_results)
    full_mem_df = pd.read_csv(general_mem_results)
    full_time_df = pd.read_csv(general_time_results)

    df = pd.DataFrame()

    dsdf = copy.deepcopy(full_df)

    rec = pd.DataFrame()
    # change that every model gets automatically selected
    print("tuning = ", tuning)
    print("type ", type(tuning))
    print("---------------------------")
    # Filter rows where 'tuning' is False
    dsdf = dsdf.loc[dsdf["tuning"] == tuning]
    print("dsdf0 = ", dsdf)
    # Use all unique models from the 'model' column when 'tuning' is False
    dsdf = dsdf.loc[dsdf["model"].isin(dsdf['model'].unique())]
    print("dsdf1 = ", dsdf)
    # if tuning:
    #     dsdf = dsdf.loc[dsdf["tuning"] == True]
    #     tuned_names = []
    #     for name in models:
    #         tuned_names.append(name + "_tuned")
    #     dsdf = dsdf.loc[dsdf["model"].isin(tuned_names)]
    # else:
    #     dsdf = dsdf.loc[dsdf["tuning"] == False]
    #     dsdf = dsdf.loc[dsdf["model"].isin(models)]

    print("dsdf2 = ", dsdf)

    lenli = [0.0 for i in range(len(dsdf))]
    dsdf["score"] = lenli
    dsdf["ranking"] = lenli
    # datasets = len(dsdf.groupby("dataset").groups.keys())

    for j, row in dsdf.iterrows():
        try:

            mem_df_val = full_mem_df.set_index("model")
            time_df_val = full_time_df.set_index("model")

            xData = np.array([6000, 12000, 18000, 24000])
            yData = np.array(
                [
                    mem_df_val.loc[row["model"], "6000"],
                    mem_df_val.loc[row["model"], "12000"],
                    mem_df_val.loc[row["model"], "18000"],
                    mem_df_val.loc[row["model"], "24000"],
                ]
            )
            yData2 = np.array(
                [
                    time_df_val.loc[row["model"], "6000"],
                    time_df_val.loc[row["model"], "12000"],
                    time_df_val.loc[row["model"], "18000"],
                    time_df_val.loc[row["model"], "24000"],
                ]
            )

            fittedParameters = np.polyfit(xData, yData, 4)
            mem_pred_poly = np.polyval(fittedParameters, datasize)

            fittedParameters2 = np.polyfit(xData, yData2, 4)
            time_pred_poly = np.polyval(fittedParameters2, datasize)

            if datasize >= 24000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "24000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "24000"])
            elif datasize >= 18000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "18000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "18000"])
            elif datasize >= 12000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "12000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "12000"])
            elif datasize >= 6000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "6000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "6000"])
            else:
                if mem_pred_poly <= 0:
                    mem_pred = mem_df_val.loc[row["model"], "6000"]
                else:
                    mem_pred = copy.deepcopy(mem_pred_poly)
                if time_pred_poly <= 0:
                    time_pred = time_df_val.loc[row["model"], "6000"]
                else:
                    time_pred = copy.deepcopy(time_pred_poly)

            dsdf.at[j, "expected_memory_usage"] = mem_pred
            dsdf.at[j, "expected_runtime"] = time_pred

        except KeyError as e:
            print(f"Key error for model {row['model']}: {e}. Skipping to next iteration.")
            continue  # Überspringt den Rest des Schleifendurchlaufs

    dsdf["expected_runtime_normalized"] = (
        dsdf["expected_runtime"] - dsdf["expected_runtime"].min()
    ) / (dsdf["expected_runtime"].max() - dsdf["expected_runtime"].min())
    for j, row in dsdf.iterrows():
        dsdf.at[j, "score"] = (
            lam * ((1 - local) * row["global"] + local * row["local"])
            + (1 - lam) * row["error_rate"]
            + row["expected_runtime_normalized"] * 100 * time
        )

    allowed_memory_usage = (psutil.virtual_memory().total / 1024**2) * mem

    plots = []
    dataframes = []
    for metric in metrics:
        df_curr = dsdf.loc[dsdf["metric"] == metric].copy()

        df_out = pd.DataFrame(columns=["model", "ranking"])
        df_curr_group = df_curr.groupby(["model"])
        c = 0
        for key, item in df_curr_group:
            df_out.at[c, "model"] = key
            df_out.at[c, "score"] = df_curr_group.get_group(key)["score"].mean()
            df_out.at[c, "error_rate"] = df_curr_group.get_group(key)["error_rate"].mean()
            df_out.at[c, "global"] = df_curr_group.get_group(key)["global"].mean()
            df_out.at[c, "local"] = df_curr_group.get_group(key)["local"].mean()
            df_out.at[c, "expected_runtime_normalized"] = df_curr_group.get_group(key)["expected_runtime_normalized"].mean()
            df_out.at[c, "expected_runtime"] = df_curr_group.get_group(key)["expected_runtime"].mean()
            df_out.at[c, "expected_memory_usage"] = df_curr_group.get_group(key)["expected_memory_usage"].mean()
            c += 1

        df_out.sort_values(by="score", ascending=True, inplace=True)
        pos = 1
        for i, row in df_out.iterrows():
            if row["expected_memory_usage"] <= allowed_memory_usage:
                df_out.at[i, "ranking"] = pos
                pos += 1
            else:
                df_out.at[i, "ranking"] = 0

        df_out = df_out.loc[df_out["ranking"] != 0]
        dataframes.append(df_out)

    return dataframes


def dataset_specific_recommendation(
    dataset, models, metrics, tuning, lam, local, mem, time,
    general_results, general_mem_results, general_time_results, datasize
):
    """
    Generate new visual recommendations based on specified criteria.

    This function reads data from predefined summary files and generates recommendations based on various parameters such
    as models, metrics, tuning, and resource usage. It creates a score for each model based on a combination of
    fairness, error rate, memory usage, and runtime.

    Parameters
    ----------
    input_files : list of str
        List of dataset names to include in the analysis.
    models : list of str
        List of model names to consider.
    metrics : list of str
        List of metrics to consider for the analysis.
    tuning : bool
        Indicates whether hyperparameters were tuned.
    lam : float
        Lambda value used in the scoring formula.
    local : float
        Weight for local fairness in the scoring formula.
    mem : float
        Memory limit as a fraction of total available memory.
    time : float
        Weight for normalized runtime in the scoring formula.

    Returns
    -------
    list
        A list of plot objects representing the visual recommendations.

    Notes
    -----
    - The function assumes specific CSV files ('COMMUNITIES_Results.csv', 'COMMUNITIES_Memory_Results.csv',
      'COMMUNITIES_Runtime_Results.csv') exist and are formatted in a certain way.
    - It uses predefined fairness metrics and assumes the presence of global and local fairness measures, error rate,
      and runtime data in the datasets.
    - Resource usage (memory and runtime) is considered in the scoring, and models exceeding the memory limit are excluded.
    """
    metrics_needed = ["demographic_parity", "equalized_odds", "treatment_equality", "consistency"]

    caption = (
        f"Fair algorithm recommendation for {dataset}, Hyperparameters tuned: {tuning}"
    )

    # Currently hardcoded
    # Also currently only really works with one chosen dataset as input
    full_df = pd.read_csv(general_results)
    full_mem_df = pd.read_csv(general_mem_results)
    # if tuning:

    # The user has to be carefull to select the right path depending on hyperparametertuning
    full_time_df = pd.read_csv(general_time_results)
    # else:
    #     full_time_df = pd.read_csv("Summary/General_Runtime_Iteration_Results.csv")
    df = pd.DataFrame()
    i = 0

    dsdf = full_df.loc[full_df["dataset"] == dataset].copy()
    lenli = [0.0 for i in dsdf.index]
    dsdf.loc[:, "score"] = lenli
    dsdf.loc[:, "ranking"] = lenli

    if tuning:
        dsdf = dsdf.loc[dsdf["tuning"] == True]
    else:
        dsdf = dsdf.loc[dsdf["tuning"] == False]


    try:
        dsdf2 = copy.deepcopy(dsdf)

        dsdf = dsdf.loc[dsdf["local_lambda"] == local]
        dsdf = dsdf.loc[dsdf["lambda"] == lam]

        if dsdf.empty:
            dsdf = copy.deepcopy(dsdf2)
            dsdf = dsdf.loc[dsdf["local_lambda"] == 0]
            dsdf = dsdf.loc[dsdf["lambda"] == 0.5]
    except:
        dsdf = copy.deepcopy(dsdf2)
        dsdf = dsdf.loc[dsdf["local_lambda"] == 0]
        dsdf = dsdf.loc[dsdf["lambda"] == 0.5]



    dsdf = dsdf.loc[dsdf["model"].isin(models)]


    for j, row in dsdf.iterrows():
        mem_df_val = full_mem_df.set_index("model")
        time_df_val = full_time_df.set_index("model")

        xData = np.array([6000, 12000, 18000, 24000])

        try:
            yData = np.array(
                [
                    mem_df_val.loc[row["model"], "6000"],
                    mem_df_val.loc[row["model"], "12000"],
                    mem_df_val.loc[row["model"], "18000"],
                    mem_df_val.loc[row["model"], "24000"],
                ]
            )
            yData2 = np.array(
                [
                    time_df_val.loc[row["model"], "6000"],
                    time_df_val.loc[row["model"], "12000"],
                    time_df_val.loc[row["model"], "18000"],
                    time_df_val.loc[row["model"], "24000"],
                ]
            )

            fittedParameters = np.polyfit(xData, yData, 4)
            mem_pred_poly = np.polyval(fittedParameters, datasize)

            fittedParameters2 = np.polyfit(xData, yData2, 4)
            time_pred_poly = np.polyval(fittedParameters2, datasize)

            if datasize >= 24000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "24000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "24000"])
            elif datasize >= 18000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "18000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "18000"])
            elif datasize >= 12000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "12000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "12000"])
            elif datasize >= 6000:
                mem_pred = max(mem_pred_poly, mem_df_val.loc[row["model"], "6000"])
                time_pred = max(time_pred_poly, time_df_val.loc[row["model"], "6000"])
            else:
                if mem_pred_poly <= 0:
                    mem_pred = mem_df_val.loc[row["model"], "6000"]
                else:
                    mem_pred = copy.deepcopy(mem_pred_poly)
                if time_pred_poly <= 0:
                    time_pred = time_df_val.loc[row["model"], "6000"]
                else:
                    time_pred = copy.deepcopy(time_pred_poly)

            dsdf.at[j, "expected_memory_usage"] = mem_pred
            #dsdf.at[j, "expected_runtime"] = time_pred
        except KeyError as e:
            print(f"Key error for model {row['model']}: {e}. Skipping to next iteration.")
            continue  # Überspringt den Rest des Schleifendurchlaufs

    #dsdf["expected_runtime_normalized"] = (
    #    dsdf["expected_runtime"] - dsdf["expected_runtime"].min()
    #) / (dsdf["expected_runtime"].max() - dsdf["expected_runtime"].min())
    dsdf["runtime_normalized"] = (
        dsdf["runtime"] - dsdf["runtime"].min()
    ) / (dsdf["runtime"].max() - dsdf["runtime"].min())
    for j, row in dsdf.iterrows():
        dsdf.at[j, "score"] = (
            lam * ((1 - local) * row["global"] + local * row["local"])
            + (1 - lam) * row["error_rate"]
            + row["runtime_normalized"] * 100 * time
        )

    allowed_memory_usage = (psutil.virtual_memory().total / 1024**2) * mem

    df = copy.deepcopy(dsdf)

    dataframes = []
    for metric in metrics:
        df_curr = df.loc[df["metric"] == metric].copy()
        df_curr.sort_values(by="score", ascending=True, inplace=True)
        pos = 1
        for i, row in df_curr.iterrows():
            if row["expected_memory_usage"] <= allowed_memory_usage:
                df_curr.at[i, "ranking"] = pos
                pos += 1
            else:
                df_curr.at[i, "ranking"] = 0
        df_curr.drop(["dataset", "metric", "tuning", "lambda", "local_lambda"], axis=1, inplace=True)
        df_curr = df_curr.loc[df_curr["ranking"] != 0]
        dataframes.append(df_curr)

    return dataframes
