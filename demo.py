"""
Python file/script for evaluation purposes.
"""
import warnings
warnings.filterwarnings(action='ignore')
import subprocess
import time
import psutil
import os
import glob
import copy
import re
import json
import random
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

# For the experiment with two protected attributes, the dataset dictionary has to add the second attribute
# to the sens_attrs list and the corresponding value of the privileged group to the favored tuple.
DATA_DICT2 = {
    "german": {"sens_attrs": ["sex"], "label": "job", "favored": (0)},
    "compas": {"sens_attrs": ["Race"], "label": "Two_yr_Recidivism", "favored": (0)},
    "communities": {"sens_attrs": ["race"], "label": "crime", "favored": (0)},
    "credit_card_clients": {"sens_attrs": ["sex"], "label": "payment", "favored": (1)},
    "adult_data_set": {"sens_attrs": ["sex"], "label": "salary", "favored": (0)},
    "acs2017_census": {"sens_attrs": ["Race"], "label": "Income", "favored": (0)},
    "implicit30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)}
}

with open("Datasets/dataset_dict.json") as f:
    DATA_DICT = json.load(f)

NEW_DATASET = ""

MODEL_NAMES_FOR_EXPERIMENTS = [
    "Reweighing",
    "LFR",
    "DisparateImpactRemover",
    "Fair-SMOTE",
    "LTDD",
    "PrejudiceRemover",
    "FairnessConstraintModel",
    "DisparateMistreatmentModel",
    "GerryFairClassifier",
    "AdversarialDebiasing",\
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "MetaFairClassifier",
    "FAGTB",
    "FairGeneralizedLinearModel",
    "GradualCompatibility",
    "RejectOptionClassification",
    "EqOddsPostprocessing",
    "CalibratedEqOddsPostprocessing",
    "JiangNachum",
    "FaX",
]

MODEL_NAMES = [
    "Reweighing",
    "LFR",
    "DisparateImpactRemover",
    "Fair-SMOTE",
    "LTDD",
    "PrejudiceRemover",
    "FairnessConstraintModel",
    "DisparateMistreatmentModel",
    "GerryFairClassifier",
    "AdversarialDebiasing",\
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "MetaFairClassifier",
    "FAGTB",
    "FairGeneralizedLinearModel",
    "GradualCompatibility",
    "RejectOptionClassification",
    "EqOddsPostprocessing",
    "CalibratedEqOddsPostprocessing",
    "JiangNachum",
    "FaX",
]

FAIRNESS_METRICS = [
    "demographic_parity",
    "equalized_odds",
    "treatment_equality",
    "consistency"
]


PLOTTED = {
    "dataset_specific_dp_tab": False,
    "dataset_specific_eo_tab": False,
    "dataset_specific_te_tab": False,
    "dataset_specific_const_tab": False,
    "general_recommendation_dp_tab": False,
    "general_recommendation_eo_tab": False,
    "general_recommendation_te_tab": False,
    "general_recommendation_const_tab": False,
    "experimental_results_dp_tab": False,
    "experimental_results_eo_tab": False,
    "experimental_results_te_tab": False,
    "experimental_results_const_tab": False,
    "global_local_error_dp_tab": False,
    "global_local_error_eo_tab": False,
    "global_local_error_te_tab": False,
    "global_local_error_const_tab": False,
    "bias_metric_corr_dp_tab": False,
    "bias_metric_corr_eo_tab": False,
    "bias_metric_corr_te_tab": False,
    "bias_metric_corr_const_tab": False,
    "global_spm_tab": False,
    "local_spm_tab": False,
    "tuning_results_dp_tab": False,
    "tuning_results_eo_tab": False,
    "tuning_results_te_tab": False,
    "tuning_results_const_tab": False,
}


def extract_target_and_dataset(file_path):
    """
    Extract the target optimization metric and dataset name from a file path.

    Parameters
    ----------
    file_path : str
        The file path from which to extract the target optimization metric and dataset name.

    Returns
    -------
    target_optimization : str
        The target optimization metric extracted from the file path.
    dataset_name : str
        The dataset name extracted from the file path.
    """
    path_parts = file_path.split(os.path.sep)
    target_optimization = path_parts[1]
    dataset_name = path_parts[2]
    return target_optimization, dataset_name


def concatenate_evaluation_files(evaluation_file_path_list, tuning):
    """
    Concatenate multiple CSV files containing evaluation results into a single DataFrame.

    Parameters
    ----------
    evaluation_file_path_list : list of str
        List of file paths to the CSV files containing evaluation results.
        Example: ["Results/demographic_parity/communities/1/EVALUATION_communities.csv"]

    Returns
    -------
    concatenated_df : DataFrame
        A pandas DataFrame containing the concatenated evaluation results.
    """
    dfs = []
    for file in evaluation_file_path_list:
        target_opt, dataset = extract_target_and_dataset(file)
        df = pd.read_csv(file, index_col=0)
        df["target_opt"] = target_opt
        df["dataset"] = dataset
        dfs.append(df)
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df["tuning"] = tuning
    local_metric_dict = {
        "demographic_parity": "lrd_dp",
        "equalized_odds": "lrd_eod",
        "equal_opportunity": "lrd_eop",
        "treatment_equality": "lrd_te",
    }
    df = concatenated_df[
        ["dataset", "model", "target_opt", "tuning", "error_rate"]
    ].copy()
    global_list = [
        metric
        for metric in concatenated_df["target_opt"].unique()
        for metric in concatenated_df[concatenated_df["target_opt"] == metric][metric]
    ]
    local_list = [
        metric
        for metric in concatenated_df["target_opt"].unique()
        for metric in concatenated_df[concatenated_df["target_opt"] == metric][
            local_metric_dict[metric]
        ]
    ]
    df["global"] = global_list
    df["local"] = local_list
    df.rename(columns={"target_opt": "metric"}, inplace=True)
    df = df.groupby(["dataset", "model", "metric", "tuning"]).mean().reset_index()
    return df.copy()


def import_results(input_files, models, tuning, general_results):
    """
    Import and filter results based on input files, models, and tuning parameters.

    This function reads a general results CSV file and filters the data based on the specified datasets (input files),
    models, and tuning parameters. It excludes certain logistic regression models when tuning is applied.

    Parameters
    ----------
    input_files : list of str
        List of dataset names to filter the results.
    models : list of str
        List of model names to include in the results.
    tuning : bool
        Indicator of whether tuning is applied. If True, specific logistic regression models are excluded.
    general_results : str
        File path to the general results CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered results based on the specified criteria.

    Notes
    -----
    The function assumes the general results CSV file has columns named 'dataset', 'tuning', and 'model'.
    It also uses a predefined list `MODEL_NAMES` assumed to be available in the scope.
    """
    full_df = pd.read_csv(general_results)
    df = pd.concat(
        [full_df.loc[full_df["dataset"] == input_file] for input_file in input_files]
    )
    df = df.loc[df["tuning"] == tuning]

    model_list = [m for m in models if m in copy.deepcopy(MODEL_NAMES)]
    if tuning and "LogisticRegression" in model_list:
        model_list.remove("LogisticRegression")
        model_list.remove("LogisticRegressionRemoved")
    df = df.loc[df["model"].isin(model_list)]

    return df


def compress_legend(fig):
    """
    Compress the legend of a Plotly figure for better readability.

    This function modifies a Plotly figure's legend to reduce redundancy and improve clarity. It handles traces with
    combined group names, separating them and ensuring each group is represented only once in the legend.

    Parameters
    ----------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object whose legend needs to be compressed.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The modified Plotly figure with a compressed legend.
    """
    group1_base, group2_base = fig.data[0].name.split(",")
    lines_marker_name = []
    for i, trace in enumerate(fig.data):
        part1, part2 = trace.name.split(",")
        if part1 == group1_base:
            lines_marker_name.append(
                {
                    "line": trace.line.to_plotly_json(),
                    "marker": trace.marker.to_plotly_json(),
                    "mode": trace.mode,
                    "name": part2.lstrip(" "),
                }
            )
        if part2 != group2_base:
            trace["name"] = ""
            trace["showlegend"] = False
        else:
            trace["name"] = part1

    ## Add the line/markers for the 2nd group
    for lmn in lines_marker_name:
        lmn["line"]["color"] = "black"
        lmn["marker"]["color"] = "black"
        fig.add_trace(go.Scatter(y=[None], **lmn))
    fig.update_layout(
        legend_title_text="", legend_itemclick=False, legend_itemdoubleclick=False
    )
    return fig


def generate_bias_plots(
    df, input_files, metrics, tuning, test_split_size, n_iterations
):
    """
    Generate bar plots for analyzing the fairness achieved in the classifications
    in realation to bias in datasets of selected fair learning algorithms.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing fairness evaluation results.
    input_files : list of str
        List of dataset names related to bias.
    metrics : list of str
        List of fairness metrics to visualize.
    tuning : bool
        Flag indicating if hyperparameter tuning was employed.
    test_split_size : float
        Proportion of the dataset to include in the test split.
    n_iterations : int
        Number of iterations for each experiment.

    Returns
    -------
    plots : dict
        A dictionary where keys are the names of fairness metrics and the values are gr.BarPlot objects.
    """
    caption = f"Hyperparameters tuned: {tuning}, Test split size: {test_split_size}, Number of iterations: {n_iterations}"
    plots = dict({})
    for metric in metrics:
        plots[metric] = gr.BarPlot(
            df.loc[df["target_opt"] == metric],
            x="dataset",
            y=metric,
            color="model",
            vertical=True,
            group="model",
            group_title="",
            x_title="",
            tooltip=["dataset", "model", metric],
            caption=caption,
            height=150,
            width=500,
            show_actions_button=True,
        )

    plots["error_rate"] = gr.BarPlot(
        df.groupby(["dataset", "model"]).mean("error_rate").reset_index(),
        x="model",
        y="error_rate",
        color="model",
        vertical=False,
        group="dataset",
        group_title="",
        x_title="",
        tooltip=["dataset", "model", "error_rate"],
        caption=caption,
        height=150,
        width=500,
        show_actions_button=True,
    )

    return plots


def run_fairness_algorithms(
    input_files,
    models,
    metrics,
    tuning,
    test_split_size,
    n_iterations,
    lam,
    local_lam,
    opt=False,
):
    """
    Run fairness algorithms across multiple datasets, models, and metrics.

    This function iterates over provided datasets, models, and metrics to execute fairness algorithms. It handles
    directory creation for results, executes the main algorithm, and then evaluates the results. Option for
    hyperparameter tuning is included.

    Parameters
    ----------
    input_files : list of str
        List of dataset names to be analyzed.
    models : list of str
        List of models to be used in the analysis.
    metrics : list of str
        List of fairness metrics to be evaluated.
    tuning : bool
        Indicates whether hyperparameter tuning is applied.
    test_split_size : float
        The size of the test split.
    n_iterations : int
        The number of iterations per model.
    lam : float
        Lambda value for the fairness algorithm.
    local_lam : float
        Local lambda value for the fairness algorithm.
    opt : bool, optional
        Indicates whether optimization is applied, by default False.
    falcc : bool, optional
        Indicates whether to use the FALCC algorithm, by default False.

    Notes
    -----
    - The function uses subprocesses to call external Python scripts (`main.py` and `evaluation.py`).
    - It relies on a predefined data dictionary (`DATA_DICT`) for dataset-specific parameters.
    """
    df_new = pd.DataFrame()
    pos = 0
    df = pd.read_csv("Summary/General_Results.csv")
    local_metric_dict = {
        "demographic_parity": "lrd_dp",
        "equalized_odds": "lrd_eod",
        "equal_opportunity": "lrd_eop",
        "treatment_equality": "lrd_te",
        "consistency": "consistency",
    }
    memory_usage = []
    for input_file in input_files:
        sensitive = DATA_DICT[input_file]["sens_attrs"]
        label = DATA_DICT[input_file]["label"]
        favored = DATA_DICT[input_file]["favored"]

        for metric in metrics:
            for randomstate in range(1, n_iterations + 1):
                if randomstate == 1:
                    pos_old = copy.deepcopy(pos)
                else:
                    pos = copy.deepcopy(pos_old)
                rs = random.randint(1,1000)
                link = (
                    "Results/"
                    + str(metric)
                    + "/"
                    + str(input_file)
                    + "/"
                    + str(rs)
                    + "/"
                )
                try:
                    os.makedirs(link)
                except FileExistsError:
                    pass

                try:
                    mem_summary = pd.read_csv(link + "MEMORY_USAGE.csv")
                    count = len(mem_summary)
                except:
                    mem_summary = pd.DataFrame()
                    count = 0

                for model in models:
                    mem_summary.at[count, "dataset"] = input_file
                    mem_summary.at[count, "model"] = model
                    mem_summary.at[count, "metric"] = metric
                    try:
                        subprocess.check_call(
                            [
                                "mprof",
                                "run",
                                "-C",
                                "-M",
                                "main.py",
                                "--output",
                                str(link),
                                "--ds",
                                str(input_file),
                                "--sensitive",
                                str(sensitive),
                                "--favored",
                                str(favored),
                                "--label",
                                str(label),
                                "--testsize",
                                str(test_split_size),
                                "--models",
                                str([model]),
                                "--metric",
                                str(metric),
                                "--randomstate",
                                str(rs),
                                "--tuning",
                                str(tuning),
                                "--opt",
                                str(opt),
                                "--lam",
                                str(lam),
                            ]
                        )

                        stdout = subprocess.check_output(["mprof", "peak"]).decode(
                            "utf-8"
                        )

                        linesplit = re.split("\n", stdout)
                        #max_mem = re.sub("\n", "", re.split("\t", linesplit[1])[-1])
                        #mem_summary.at[count, "max_mem"] = re.sub(r"\sMiB", "", str(max_mem))

                        max_mem = float(re.findall("\d+\.\d+", re.split("\t", linesplit[1])[-1])[0])
                        mem_summary.at[count, "max_mem"] = max_mem

                    except Exception as e:
                        mem_summary.at[count, "max_mem"] = e

                    count += 1
                    mem_summary.to_csv(link + "MEMORY_USAGE.csv", index=False)

                if tuning:
                    model_list_eval = []
                    for model in models:
                        model_list_eval.append(model + "_tuned")
                    model_list_eval.sort()
                else:
                    model_list_eval = models


                subprocess.check_call(
                    [
                        "python",
                        "-Wignore",
                        "evaluation.py",
                        "--folder",
                        str(link),
                        "--ds",
                        str(input_file),
                        "--sensitive",
                        str(sensitive),
                        "--favored",
                        str(favored),
                        "--label",
                        str(label),
                        "--models",
                        str(model_list_eval),
                        "--metric",
                        str(metric),
                    ]
                )

                # memory_usage_df = pd.DataFrame(memory_usage)
                # memory_usage_df.to_csv("Results/memory_usage.csv", index=False)

                csv = pd.read_csv(link + "EVALUATION_" + str(input_file) + ".csv", index_col="model")
                
                for model in models:
                    if tuning:
                        modelname = model + "_tuned"
                    else:
                        modelname = model
                    res = df.loc[(df['model'] == model)
                        & (df['dataset'] == input_file)
                        & (df['metric'] == metric)
                        & (df['tuning'] == tuning)
                        & (df['lambda'] == lam)
                        & (df['local_lambda'] == local_lam)]
                    if res.empty:
                        pos = len(df)
                        df.at[pos, 'model'] = model
                        df.at[pos, 'dataset'] = input_file
                        df.at[pos, 'metric'] = metric
                        df.at[pos, 'tuning'] = tuning
                        df.at[pos, 'lambda'] = lam
                        df.at[pos, 'local_lambda'] = local_lam
                        df.at[pos, 'count'] = 1
                        df.at[pos, 'error_rate'] = csv.at[modelname, 'error_rate']
                        df.at[pos, 'global'] = csv.at[modelname, metric]
                        df.at[pos, 'local'] = csv.at[modelname, local_metric_dict[metric]]
                    else:
                        for i, row in res.iterrows():
                            counter = df.loc[i, 'count']
                            df.loc[i, 'error_rate'] = round((csv.at[modelname, 'error_rate'] + df.loc[i, 'error_rate'] * counter)/(counter + 1), 3)
                            df.loc[i, 'global'] = round((csv.at[modelname, metric] + df.loc[i, 'global'] * counter)/(counter + 1), 3)
                            df.loc[i, 'local'] = round((csv.at[modelname, local_metric_dict[metric]] + df.loc[i, 'local'] * counter)/(counter + 1), 3)
                            df.loc[i, 'count'] += 1

                    if randomstate == 1:
                        df_new.at[pos, 'model'] = model
                        df_new.at[pos, 'dataset'] = input_file
                        df_new.at[pos, 'metric'] = metric
                        df_new.at[pos, 'error_rate'] = csv.at[modelname, 'error_rate']
                        df_new.at[pos, 'global'] = csv.at[modelname, metric]
                        df_new.at[pos, 'local'] = csv.at[modelname, local_metric_dict[metric]]
                        df_new.at[pos, 'tuning'] = tuning
                        df_new.at[pos, 'lambda'] = lam
                        df_new.at[pos, 'local_lambda'] = local_lam
                        df_new.at[pos, 'count'] = 1
                    else:
                        counter = df_new.at[pos, 'count']
                        df_new.at[pos, 'error_rate'] = round((csv.at[modelname, 'error_rate'] + df_new.at[pos, 'error_rate'] * counter)/(counter+1), 3)
                        df_new.at[pos, 'global'] = round((csv.at[modelname, metric] + df_new.at[pos, 'global'] * counter)/(counter+1), 3)
                        df_new.at[pos, 'local'] = round((csv.at[modelname, local_metric_dict[metric]] + df_new.at[pos, 'local'] * counter)/(counter+1), 3)
                        df_new.at[pos, 'count'] += 1
                    pos += 1

    df_new.to_csv("Results/complete_results.csv", index=False)
    df.to_csv("Summary/General_Results.csv", index=False)

    return [gr.CheckboxGroup(visible=False), df_new]



def generate_general_recommendation_plots(
    models,
    metrics,
    tuning,
    lam,
    local,
    datasize,
    mem,
    time,
    general_results="Summary/General_Results.csv",
    general_mem_results="Summary/General_Memory_Results.csv",
    general_time_results="Summary/General_Runtime_Results.csv",
):
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
    general_results : str, optional
        Path to the CSV file containing general results, by default "Summary/General_Results_New.csv".
    general_mem_results : str, optional
        Path to the CSV file containing general memory results, by default "Summary/General_Memory_Results.csv".
    general_time_results : str, optional
        Path to the CSV file containing general runtime results, by default "Summary/General_Runtime_Results.csv".

    Returns:
    --------
    list
        A list of plotly bar plots and dataframes, one for each metric.
    """
    metrics_needed = ["demographic_parity", "equalized_odds", "treatment_equality", "consistency"]

    caption = f"General recommendation, Hyperparameters tuned: {tuning}"

    full_df = pd.read_csv("Summary/General_Results.csv")
    full_mem_df = pd.read_csv("Summary/General_Memory_Results.csv")
    if tuning:
        full_time_df = pd.read_csv("Summary/General_Runtime_Results.csv")
    else:
        full_time_df = pd.read_csv("Summary/General_Runtime_Iteration_Results.csv")
    df = pd.DataFrame()
    dsdf = copy.deepcopy(full_df)
    rec = pd.DataFrame()
    if tuning:
        dsdf = dsdf.loc[dsdf["tuning"] == True]
    else:
        dsdf = dsdf.loc[dsdf["tuning"] == False]
    if "ALL" not in models:
        dsdf = dsdf.loc[dsdf["model"].isin(models)]
    else:
        model_list = copy.deepcopy(MODEL_NAMES)
        model_list.remove("ALL")
        if tuning:
            if "LogisticRegression" in model_list:
                model_list.remove("LogisticRegression")
                model_list.remove("LogisticRegressionRemoved")
            dsdf = dsdf.loc[dsdf["model"].isin(model_list)]
    lenli = [0.0 for i in range(len(dsdf))]
    dsdf["score"] = lenli
    dsdf["ranking"] = lenli
    # datasets = len(dsdf.groupby("dataset").groups.keys())
    for j, row in dsdf.iterrows():
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
    for metric in metrics_needed:
        try:
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
            dataframes.append(
                gr.Dataframe(df_out, headers=list(df_out.columns), height=300)
            )

            plots.append(
                px.bar(
                    df_out,
                    x="model",
                    y="score",
                    hover_data={
                        "model": True,
                        "global": ":.2f",
                        "local": ":.2f",
                        "error_rate": ":.2f",
                        "score": ":.2f",
                        "expected_memory_usage": ":.2f",
                        "expected_runtime": ":.2f",
                    },
                    labels={"model": "fairness algorithm"},
                    title=caption,
                )
            )
        except:
            dataframes.append(gr.Dataframe())
            plots.append(px.bar())

    return plots + dataframes


def generate_general_fairness_plots(
    input_files, models, metrics, tuning, general_results="Summary/General_Results.csv"
):
    """
    Generate general fairness plots for a set of input files, models, and metrics.

    Parameters
    ----------
    input_files : list of str
        List of input file paths.
    models : list of str
        List of fairness algorithm names.
    metrics : list of str
        List of metric names.
    tuning : bool
        Whether or not hyperparameters were tuned.
    general_results : str, optional
        Path to the general results CSV file, by default "Summary/General_Results.csv".

    Returns
    -------
    list of px.bar
        List of plotly bar plots containing the generated plots.
    """
    metrics_needed = ["demographic_parity", "equalized_odds", "treatment_equality", "consistency"]
    df = import_results(input_files, models, tuning, general_results)
    plots = []
    for metric in metrics_needed:
        if metric in metrics:
            for plot_type in ["global", "local", "error_rate"]:
                plots.append(
                    px.bar(
                        df.loc[df["metric"] == metric],
                        x="model",
                        y=plot_type,
                        color="dataset",
                        barmode="group",
                        hover_data={"dataset": True, "model": True, plot_type: ":.2f"},
                        labels={"model": "fairness algorithm"},
                        title=f"{plot_type} {metric} results for datasets \nHyperparameters tuned: {tuning}",
                    )
                )
        else:
            for plot_type in ["global", "local", "error_rate"]:
                plots.append(px.bar())
    return plots


def global_local_error_scatter(
    input_files, models, metrics, tuning, general_results="Summary/General_Results.csv"
):
    """
    Generate scatter plots of local and error rate metrics against global metrics for given input files, models, and metrics.

    Parameters:
    -----------
    input_files : list of str
        List of file paths to input files.
    models : list of str
        List of model names.
    metrics : list of str
        List of metric names.
    tuning : bool
        Whether or not hyperparameters were tuned.
    general_results : str, optional
        Path to general results CSV file. Default is "Summary/General_Results.csv".

    Returns:
    --------
    plots : list of px.scatter
        List of scatter plots of local and error rate metrics against global metrics for given input files, models, and metrics.
    """
    metrics_needed = ["demographic_parity", "equalized_odds", "treatment_equality", "consistency"]
    df = import_results(input_files, models, tuning, general_results)
    plots = []
    for metric in metrics_needed:
        if metric in metrics:
            try:
                for plot_type in ["local", "error_rate"]:
                    plots.append(
                        compress_legend(
                            px.scatter(
                                df[df["metric"] == metric],
                                x="global",
                                y=plot_type,
                                color="dataset",
                                symbol="model",
                                symbol_sequence=[
                                    "circle",
                                    "square",
                                    "diamond",
                                    "cross",
                                    "x",
                                    "triangle-up",
                                    "triangle-down",
                                    "triangle-left",
                                    "triangle-right",
                                    "triangle-ne",
                                    "triangle-se",
                                    "triangle-sw",
                                    "triangle-nw",
                                    "pentagon",
                                    "hexagon",
                                    "hexagon2",
                                    "octagon",
                                    "star",
                                    "hexagram",
                                    "star-triangle-up",
                                    "star-triangle-down",
                                    "star-square",
                                    "star-diamond",
                                    "diamond-tall",
                                    "diamond-wide",
                                    "hourglass",
                                    "bowtie",
                                    "circle-cross",
                                    "circle-x",
                                    "square-cross",
                                    "square-x",
                                    "diamond-cross",
                                    "diamond-x",
                                    "cross-thin",
                                    "x-thin",
                                    "asterisk",
                                    "hash",
                                    "y-up",
                                    "y-down",
                                    "y-left",
                                    "y-right",
                                    "line-ew",
                                    "line-ns",
                                    "line-ne",
                                    "line-nw",
                                    "arrow-up",
                                    "arrow-down",
                                    "arrow-left",
                                    "arrow-right",
                                    "arrow-bar-up",
                                    "arrow-bar-down",
                                    "arrow-bar-left",
                                    "arrow-bar-right",
                                ],
                                hover_data={
                                    "dataset": True,
                                    "model": True,
                                    "global": ":.2f",
                                    "error_rate": ":.2f",
                                    "local": ":.2f",
                                    "tuning": True,
                                },
                                title=f"{plot_type} {metric} to global {metric}<br>Hyperparameters tuned: {tuning}",
                            )
                        )
                    )
            except:
                for plot_type in ["local", "error_rate"]:
                    plots.append(px.scatter())
        else:
            for plot_type in ["local", "error_rate"]:
                plots.append(px.scatter())

    return plots


def metric_spm(
    input_files, models, metrics, tuning, general_results="Summary/General_Results.csv"
):
    """
    Generate scatterplots of global and local bias metric correlations.

    Parameters
    ----------
    input_files : list of str
        List of file names of the input data files.
    models : list of str
        List of model names to include in the analysis.
    metrics : list of str
        List of bias metrics to include in the analysis.
    tuning : bool
        Whether or not hyperparameters were tuned.
    general_results : str, optional
        File path to the general results CSV file, by default "Summary/General_Results.csv".

    Returns
    -------
    List of px.scatter_matrix
        List of scatterplot matrices, one for global bias metrics and one for local bias metrics.
    """
    if general_results == "Summary/General_Results.csv":
        metrics = ["demographic_parity", "equalized_odds", "treatment_equality", "consistency"]
    caption = f"Scatterplots of different global bias metric correlations, Hyperparameters tuned: {tuning}"

    df = import_results(input_files, models, tuning, general_results)

    new_df = df.pivot(
        index=["dataset", "model"], columns="metric", values=["global", "local"]
    )
    new_df.columns = [f"{col[1]}_{col[0]}" for col in new_df.columns]
    new_df = new_df.reset_index()

    caption = f"Scatterplots of different global bias metric correlations, Hyperparameters tuned: {tuning}"
    try:
        global_metric_spm = px.scatter_matrix(
            new_df,
            dimensions=[f"{metric}_global" for metric in metrics],
            color="dataset",
            symbol="model",
            symbol_sequence=[
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "triangle-down",
                "triangle-left",
                "triangle-right",
                "triangle-ne",
                "triangle-se",
                "triangle-sw",
                "triangle-nw",
                "pentagon",
                "hexagon",
                "hexagon2",
                "octagon",
                "star",
                "hexagram",
                "star-triangle-up",
                "star-triangle-down",
                "star-square",
                "star-diamond",
                "diamond-tall",
                "diamond-wide",
                "hourglass",
                "bowtie",
                "circle-cross",
                "circle-x",
                "square-cross",
                "square-x",
                "diamond-cross",
                "diamond-x",
                "cross-thin",
                "x-thin",
                "asterisk",
                "hash",
                "y-up",
                "y-down",
                "y-left",
                "y-right",
                "line-ew",
                "line-ns",
                "line-ne",
                "line-nw",
                "arrow-up",
                "arrow-down",
                "arrow-left",
                "arrow-right",
                "arrow-bar-up",
                "arrow-bar-down",
                "arrow-bar-left",
                "arrow-bar-right",
            ],
            title=caption,
            hover_data=[
                "model",
                "dataset",
            ],
        )

        local_metric_spm = px.scatter_matrix(
            new_df,
            dimensions=[f"{metric}_local" for metric in metrics],
            color="dataset",
            symbol="model",
            symbol_sequence=[
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "triangle-down",
                "triangle-left",
                "triangle-right",
                "triangle-ne",
                "triangle-se",
                "triangle-sw",
                "triangle-nw",
                "pentagon",
                "hexagon",
                "hexagon2",
                "octagon",
                "star",
                "hexagram",
                "star-triangle-up",
                "star-triangle-down",
                "star-square",
                "star-diamond",
                "diamond-tall",
                "diamond-wide",
                "hourglass",
                "bowtie",
                "circle-cross",
                "circle-x",
                "square-cross",
                "square-x",
                "diamond-cross",
                "diamond-x",
                "cross-thin",
                "x-thin",
                "asterisk",
                "hash",
                "y-up",
                "y-down",
                "y-left",
                "y-right",
                "line-ew",
                "line-ns",
                "line-ne",
                "line-nw",
                "arrow-up",
                "arrow-down",
                "arrow-left",
                "arrow-right",
                "arrow-bar-up",
                "arrow-bar-down",
                "arrow-bar-left",
                "arrow-bar-right",
            ],
            title=caption,
            hover_data=[
                "model",
                "dataset",
            ],
        )
    except:
        metrics = ["demographic_parity", "equalized_odds", "treatment_equality"]
        global_metric_spm = px.scatter_matrix(
            new_df,
            dimensions=[f"{metric}_global" for metric in metrics],
            color="dataset",
            symbol="model",
            symbol_sequence=[
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "triangle-down",
                "triangle-left",
                "triangle-right",
                "triangle-ne",
                "triangle-se",
                "triangle-sw",
                "triangle-nw",
                "pentagon",
                "hexagon",
                "hexagon2",
                "octagon",
                "star",
                "hexagram",
                "star-triangle-up",
                "star-triangle-down",
                "star-square",
                "star-diamond",
                "diamond-tall",
                "diamond-wide",
                "hourglass",
                "bowtie",
                "circle-cross",
                "circle-x",
                "square-cross",
                "square-x",
                "diamond-cross",
                "diamond-x",
                "cross-thin",
                "x-thin",
                "asterisk",
                "hash",
                "y-up",
                "y-down",
                "y-left",
                "y-right",
                "line-ew",
                "line-ns",
                "line-ne",
                "line-nw",
                "arrow-up",
                "arrow-down",
                "arrow-left",
                "arrow-right",
                "arrow-bar-up",
                "arrow-bar-down",
                "arrow-bar-left",
                "arrow-bar-right",
            ],
            title=caption,
            hover_data=[
                "model",
                "dataset",
            ],
        )

        local_metric_spm = px.scatter_matrix(
            new_df,
            dimensions=[f"{metric}_local" for metric in metrics],
            color="dataset",
            symbol="model",
            symbol_sequence=[
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "triangle-down",
                "triangle-left",
                "triangle-right",
                "triangle-ne",
                "triangle-se",
                "triangle-sw",
                "triangle-nw",
                "pentagon",
                "hexagon",
                "hexagon2",
                "octagon",
                "star",
                "hexagram",
                "star-triangle-up",
                "star-triangle-down",
                "star-square",
                "star-diamond",
                "diamond-tall",
                "diamond-wide",
                "hourglass",
                "bowtie",
                "circle-cross",
                "circle-x",
                "square-cross",
                "square-x",
                "diamond-cross",
                "diamond-x",
                "cross-thin",
                "x-thin",
                "asterisk",
                "hash",
                "y-up",
                "y-down",
                "y-left",
                "y-right",
                "line-ew",
                "line-ns",
                "line-ne",
                "line-nw",
                "arrow-up",
                "arrow-down",
                "arrow-left",
                "arrow-right",
                "arrow-bar-up",
                "arrow-bar-down",
                "arrow-bar-left",
                "arrow-bar-right",
            ],
            title=caption,
            hover_data=[
                "model",
                "dataset",
            ],
        )


    for spm in [global_metric_spm, local_metric_spm]:
        spm.update_traces(diagonal_visible=False, showupperhalf=False)
        spm.update_layout(
            height=800,
        )

    return [global_metric_spm, local_metric_spm]


def tuning_result_plots(
    input_files, models, metrics, tuning, general_results="Summary/General_Results.csv"
):
    """
    Generate plots to compare the effects of hyperparameter tuning.

    This function creates a series of plots to demonstrate the differences in performance metrics (error rate, global,
    and local fairness) between tuned and untuned models for given datasets, models, and metrics.

    Parameters
    ----------
    input_files : list of str
        List of dataset names to include in the analysis.
    models : list of str
        List of model names to consider.
    metrics : list of str
        List of performance metrics to analyze.
    tuning : bool
        Indicates whether to consider hyperparameter tuning in the analysis.
    general_results : str, optional
        File path to the general results CSV file, by default "Summary/General_Results.csv".

    Returns
    -------
    list
        A list of plot objects showing the differences in metrics due to tuning.

    Notes
    -----
    - The function utilizes `import_results` to fetch tuned and untuned results.
    - It calculates the differences in error rate, global fairness, and local fairness between tuned and untuned models.
    - Plots are generated using `px.bar` for each metric and difference type.
    """
    caption = f"Plot of the effect of hyperparameter tuning"
    plots = []
    df_tuned = import_results(
        input_files=input_files,
        models=models,
        tuning=True,
        general_results=general_results,
    )
    df_untuned = import_results(
        input_files=input_files,
        models=models,
        tuning=False,
        general_results=general_results,
    )
    df_diff = pd.DataFrame(
        columns=[
            "dataset",
            "metric",
            "model",
            "error_rate_diff",
            "global_diff",
            "local_diff",
        ]
    )

    # Calculate the difference in error rate, global, and local for each model, dataset and metric
    for index, row_tuned in df_tuned.iterrows():
        dataset = row_tuned["dataset"]
        metric = row_tuned["metric"]
        model = row_tuned["model"]
        error_rate_tuned = row_tuned["error_rate"]
        global_tuned = row_tuned["global"]
        local_tuned = row_tuned["local"]

        row_untuned = df_untuned[
            (df_untuned["dataset"] == dataset)
            & (df_untuned["metric"] == metric)
            & (df_untuned["model"] == model)
        ]
        if not row_untuned.empty:
            error_rate_untuned = row_untuned["error_rate"].values[0]
            global_untuned = row_untuned["global"].values[0]
            local_untuned = row_untuned["local"].values[0]

            error_rate_diff = error_rate_untuned - error_rate_tuned
            global_diff = global_untuned - global_tuned
            local_diff = local_untuned - local_tuned

            df_diff = pd.merge(
                df_diff,
                pd.DataFrame(
                    {
                        "dataset": [dataset],
                        "metric": [metric],
                        "model": [model],
                        "error_rate_diff": [error_rate_diff],
                        "global_diff": [global_diff],
                        "local_diff": [local_diff],
                    }
                ),
                how="outer",
            )

    # Plot part
    for metric in metrics:
        for plot_type in ["global_diff", "local_diff", "error_rate_diff"]:
            plots.append(
                px.bar(
                    df_diff.loc[df_diff["metric"] == metric],
                    x="model",
                    y=plot_type,
                    color="dataset",
                    barmode="group",
                    hover_data={"dataset": True, "model": True, plot_type: ":.2f"},
                    labels={"model": "fairness algorithm"},
                    title=f"difference in {plot_type} {metric} results for datasets",
                )
            )
    return plots


def dataset_specific_recommendation(
    dataset,
    models,
    metrics,
    tuning,
    lam,
    local,
    mem,
    time,
    general_results="Summary/General_Results.csv",
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
    full_mem_df = pd.read_csv("Summary/General_Memory_Results.csv")
    if tuning:
        full_time_df = pd.read_csv("Summary/General_Runtime_Results.csv")
    else:
        full_time_df = pd.read_csv("Summary/General_Runtime_Iteration_Results.csv")
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
    datasize = len(pd.read_csv("Datasets/" + dataset + ".csv"))

    for j, row in dsdf.iterrows():
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

    df = copy.deepcopy(dsdf)

    plots = []
    for metric in metrics_needed:
        if metric in metrics:
            try:
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
                plots.append(
                    px.bar(
                        df_curr,
                        x="model",
                        y="score",
                        hover_data={
                            "model": True,
                            "global": ":.2f",
                            "local": ":.2f",
                            "error_rate": ":.2f",
                            "score": ":.2f",
                            "expected_memory_usage": ":.2f",
                            "expected_runtime": ":.2f",
                        },
                        labels={"model": "fairness algorithm"},
                        title=caption,
                    )
                )
                plots.append(gr.Dataframe(df_curr, headers=list(df_curr.columns), height=300))
            except:
                plots.append(px.bar())
                plots.append(gr.Dataframe())
        else:
            plots.append(px.bar())
            plots.append(gr.Dataframe())
    return plots


def upload_file(file):
    global NEW_DATASET
    df = pd.read_csv(file.name)
    NEW_DATASET = re.split("/", file.name)[-1]
    NEW_DATASET = NEW_DATASET.split("\\")[-1]
    df.to_csv("Datasets/" + NEW_DATASET, index=False)
    return file.name


def insert_dataset(file_output, prot_attr, label):
    global DATA_DICT
    ds_attr = dict()
    ds_attr["sens_attrs"] = [prot_attr]
    ds_attr["label"] = label
    ds_attr["favored"] = 1
    dataname = re.sub(r"\.csv", "", NEW_DATASET)
    DATA_DICT[dataname] = ds_attr
    with open("Datasets/dataset_dict.json", "w+") as f:
        json.dump(DATA_DICT, f)


def display_models(radio):
    """
    Display model selection options as a checkbox group.
    """
    if radio == "ALL":
        return gr.CheckboxGroup(MODEL_NAMES, value=MODEL_NAMES, visible=False)
    else:
        return gr.CheckboxGroup(MODEL_NAMES, value=[], visible=True)


def refresh_plots(evt: gr.SelectData):
    """
    Refresh plots in a graphical interface upon a specific trigger event.

    This function updates the display of plots based on user interactions. It checks if the plot associated with
    the event's target has already been plotted. If not, it marks it as plotted and refreshes the display accordingly.
    This is needed so the plots that were first hidden can be displayed correctly.

    Parameters
    ----------
    evt : gr.SelectData
        The event data from the user interaction, containing information about the target plot.

    Returns
    -------
    list or None
        Returns a list of plot children if multiple plots need updating, a single plot child if only one plot needs
        updating, or None if no update is needed.

    Notes
    -----
    - The function uses a global variable 'PLOTTED' to track the state of plot rendering.
    - The structure of 'evt.target.children' is important for determining the return value.
    """
    global PLOTTED
    if not PLOTTED[evt.target.id]:
        PLOTTED[evt.target.id] = True
        if len(evt.target.children[0].children) == 1:
            return None
        else:
            return [None] * len(evt.target.children[0].children)
    else:
        if len(evt.target.children[0].children) == 1:
            return evt.target.children[0].children[0]
        else:
            return evt.target.children[0].children


def create_fairness_metric_tab(metric_name, metric_id):
    """
    Create a tab layout for a specific fairness metric in a graphical interface.

    This function generates a tab for a given fairness metric, with separate plots for global and local assessments
    of the metric, as well as a plot for the error rate.

    Parameters
    ----------
    metric_name : str
        The name of the fairness metric.
    metric_id : str
        The unique identifier for the metric tab.

    Returns
    -------
    tuple
        A tuple containing the metric tab and plot objects for global metric, local metric, and error rate.

    Notes
    -----
    - The function uses `gr.Tab` and `gr.Column` for layout and `gr.Plot` for plot creation.
    - The plots are initially empty and are intended to be populated later based on specific data.
    """
    with gr.Tab(metric_name, id=metric_id) as metric_tab:
        with gr.Column():
            global_metric_plot = gr.Plot(
                label=f"{metric_name} (global)", show_label=False
            )
            local_metric_plot = gr.Plot(
                label=f"{metric_name} (local)", show_label=False
            )
            error_metric_plot = gr.Plot(label="Error Rate")
        return metric_tab, global_metric_plot, local_metric_plot, error_metric_plot


with gr.Blocks(
    theme=gr.themes.Default(
        text_size="lg",
        spacing_size="sm",
    )
) as fairness_dashboard:
    gr.Markdown(
        """
        # FairCR – an evaluation and recommendation system for fair classifications
        """
    )
    with gr.Tab("Algorithm Recommendation"):
        gr.Markdown(
            """
            Select your preferences to get a recommendations for fairness algorithms.
            """
        )
        with gr.Column():
            all_selection = gr.Radio(
                ["ALL", "Choose models"], value="ALL", label="Models"
            )
            selected_models = gr.CheckboxGroup(
                MODEL_NAMES, value=MODEL_NAMES, label="Models", visible=False
            )
            all_selection.input(display_models, [all_selection], [selected_models])
            alt_time_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0,
                label="Runtime weight",
            )
            alt_mem_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Maximum usage of available memory (in %)",
            )

            alt_lambda_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.5,
                label="Global fairness weight",
            )
            alt_local_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0,
                label="Local fairness weight",
            )

            is_tuning_enabled = gr.Checkbox(label="Tune hyperparameters?")
            with gr.Row(visible=False):
                selected_fairness_metrics = gr.CheckboxGroup(
                    FAIRNESS_METRICS,
                    value=FAIRNESS_METRICS,
                    label="Target Fairness Metric",
                    visible=True,
                )

            with gr.Tab("General Recommendations"):
                with gr.Column():
                    datasize_slider = gr.Slider(
                        minimum=1000,
                        maximum=100000,
                        step=1000,
                        value=10000,
                        label="Expected size of the dataset (in samples)",
                    )
                    start_button_general_recommendation = gr.Button(
                        "Show/Refresh", variant="primary"
                    )
                    with gr.Tab(
                        "Demographic Parity", id="general_recommendation_dp_tab"
                    ) as general_recommendation_dp_tab:
                        with gr.Row():
                            alt_score_demographic_parity = gr.Plot(
                                label="Demographic Parity", show_label=False
                            )
                        with gr.Row():
                            alt_table_demographic_parity = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Equalized Odds", id="general_recommendation_eo_tab"
                    ) as general_recommendation_eo_tab:
                        with gr.Row():
                            alt_score_equalized_odds = gr.Plot(
                                label="Equalized Odds", show_label=False
                            )
                        with gr.Row():
                            alt_table_equalized_odds = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Treatment Equality", id="general_recommendation_te_tab"
                    ) as general_recommendation_te_tab:
                        with gr.Row():
                            alt_score_treatment_equality = gr.Plot(
                                label="Treatment Equality", show_label=False
                            )
                        with gr.Row():
                            alt_table_treatment_equality = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Consistency", id="general_recommendation_const_tab"
                    ) as general_recommendation_const_tab:
                        with gr.Row():
                            alt_score_consistency = gr.Plot(
                                label="Consistency", show_label=False
                            )
                        with gr.Row():
                            alt_table_consistency = gr.Dataframe(type="pandas")

                generate_general_recommendation_plots_input = [
                    selected_models,
                    selected_fairness_metrics,
                    is_tuning_enabled,
                    alt_lambda_slider,
                    alt_local_slider,
                    datasize_slider,
                    alt_mem_slider,
                    alt_time_slider,
                ]
                generate_general_recommendation_plots_output = [
                    alt_score_demographic_parity,
                    alt_score_equalized_odds,
                    alt_score_treatment_equality,
                    alt_score_consistency,
                    alt_table_demographic_parity,
                    alt_table_equalized_odds,
                    alt_table_treatment_equality,
                    alt_table_consistency,
                ]

                for tab in [
                    general_recommendation_dp_tab,
                    general_recommendation_eo_tab,
                    general_recommendation_te_tab,
                    general_recommendation_const_tab,
                ]:
                    tab.select(
                        refresh_plots,
                        inputs=[],
                        outputs=[tab.children[0].children[0]],
                    ).then(
                        generate_general_recommendation_plots,
                        generate_general_recommendation_plots_input,
                        generate_general_recommendation_plots_output,
                    )

                start_button_general_recommendation.click(
                    generate_general_recommendation_plots,
                    generate_general_recommendation_plots_input,
                    generate_general_recommendation_plots_output,
                )

            with gr.Tab("Dataset-specific Recommendations"):
                with gr.Column():
                    selected_datasets = gr.Radio(DATA_DICT.keys(), label="Datasets")
                    start_button_recommendation = gr.Button(
                        "Show/Refresh", variant="primary"
                    )
                    with gr.Tab(
                        "Demographic Parity", id="dataset_specific_dp_tab"
                    ) as dataset_specific_dp_tab:
                        with gr.Row():
                            score_demographic_parity = gr.Plot(
                                label="Demographic Parity", show_label=False
                            )
                        with gr.Row():
                            table_demographic_parity = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Equalized Odds", id="dataset_specific_eo_tab"
                    ) as dataset_specific_eo_tab:
                        with gr.Row():
                            score_equalized_odds = gr.Plot(
                                label="Equalized Odds", show_label=False
                            )
                        with gr.Row():
                            table_equalized_odds = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Treatment Equality", id="dataset_specific_te_tab"
                    ) as dataset_specific_te_tab:
                        with gr.Row():
                            score_treatment_equality = gr.Plot(
                                label="Treatment Equality", show_label=False
                            )
                        with gr.Row():
                            table_treatment_equality = gr.Dataframe(type="pandas")
                    with gr.Tab(
                        "Consistency", id="dataset_specific_const_tab"
                    ) as dataset_specific_const_tab:
                        with gr.Row():
                            score_consistency = gr.Plot(
                                label="Consistency", show_label=False
                            )
                        with gr.Row():
                            table_consistency = gr.Dataframe(type="pandas")

                    new_visuals_recommendation_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                        alt_lambda_slider,
                        alt_local_slider,
                        alt_mem_slider,
                        alt_time_slider,
                    ]

                    new_visuals_recommendation_output = [
                        score_demographic_parity,
                        table_demographic_parity,
                        score_equalized_odds,
                        table_equalized_odds,
                        score_treatment_equality,
                        table_treatment_equality,
                        score_consistency,
                        table_consistency,
                    ]

                    for tab in [
                        dataset_specific_dp_tab,
                        dataset_specific_eo_tab,
                        dataset_specific_te_tab,
                        dataset_specific_const_tab,
                    ]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=[tab.children[0].children[0]],
                        ).then(
                            dataset_specific_recommendation,
                            new_visuals_recommendation_input,
                            new_visuals_recommendation_output,
                        )

                    start_button_recommendation.click(
                        dataset_specific_recommendation,
                        new_visuals_recommendation_input,
                        new_visuals_recommendation_output,
                    )

    with gr.Tab("Evaluation of Fair Classification Algorithms"):
        gr.Markdown(
            """
            Select datasets, algorithm and target metric to start a comparison.
            """
        )
        with gr.Column():
            with gr.Column() as fairness_input_panel:
                selected_datasets = gr.CheckboxGroup(DATA_DICT.keys(), label="Datasets")
                all_selection = gr.Radio(
                    ["ALL", "Choose models"], value="ALL", label="Models"
                )
                selected_models = gr.CheckboxGroup(
                    MODEL_NAMES, value=MODEL_NAMES, label="Models", visible=False
                )
                all_selection.input(display_models, [all_selection], [selected_models])

                with gr.Row(visible=False):
                    selected_fairness_metrics = gr.CheckboxGroup(
                        FAIRNESS_METRICS,
                        value=FAIRNESS_METRICS,
                        label="Target Fairness Metric",
                        visible=True,
                    )
                with gr.Column():
                    is_tuning_enabled = gr.Checkbox(label="Tune hyperparameters?")

            with gr.Column() as output_panel:
                with gr.Tab("Experimental Results"):
                    with gr.Column():
                        with gr.Row():
                            start_button_general = gr.Button(
                                "Show/Refresh", variant="primary"
                            )

                        (
                            exp_dp_tab,
                            exp_dp_global_plot,
                            exp_dp_local_plot,
                            exp_dp_error_plot,
                        ) = create_fairness_metric_tab(
                            "Demographic Parity", "experimental_results_dp_tab"
                        )
                        (
                            exp_eo_tab,
                            exp_eo_global_plot,
                            exp_eo_local_plot,
                            exp_eo_error_plot,
                        ) = create_fairness_metric_tab(
                            "Equalized Odds", "experimental_results_eo_tab"
                        )
                        (
                            exp_te_tab,
                            exp_te_global_plot,
                            exp_te_local_plot,
                            exp_te_error_plot,
                        ) = create_fairness_metric_tab(
                            "Treatment Equality", "experimental_results_te_tab"
                        )
                        (
                            exp_const_tab,
                            exp_const_global_plot,
                            exp_const_local_plot,
                            exp_const_error_plot,
                        ) = create_fairness_metric_tab(
                            "Consistency", "experimental_results_const_tab"
                        )

                    generate_general_fairness_plots_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                    ]
                    generate_general_fairness_plots_output = [
                        exp_dp_global_plot,
                        exp_dp_local_plot,
                        exp_dp_error_plot,
                        exp_eo_global_plot,
                        exp_eo_local_plot,
                        exp_eo_error_plot,
                        exp_te_global_plot,
                        exp_te_local_plot,
                        exp_te_error_plot,
                        exp_const_global_plot,
                        exp_const_local_plot,
                        exp_const_error_plot,
                    ]
                    for tab in [exp_dp_tab, exp_eo_tab, exp_te_tab, exp_const_tab]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        ).then(
                            generate_general_fairness_plots,
                            generate_general_fairness_plots_input,
                            generate_general_fairness_plots_output,
                        )

                        start_button_general.click(
                            generate_general_fairness_plots,
                            generate_general_fairness_plots_input,
                            generate_general_fairness_plots_output,
                        )

                with gr.Tab("Global/Local and Global/Error Correlation"):
                    with gr.Column():
                        with gr.Row():
                            start_button_global_local_error_scatter = gr.Button(
                                "Show/Refresh", variant="primary"
                            )
                        with gr.Tab(
                            "Demographic Parity", id="global_local_error_dp_tab"
                        ) as global_local_error_dp_tab:
                            with gr.Row():
                                scatter_demographic_parity = gr.Plot(
                                    label="Demographic Parity", show_label=False
                                )
                                scatter_local_demographic_parity = gr.Plot(
                                    label="Demographic Parity", show_label=False
                                )
                        with gr.Tab(
                            "Equalized Odds", id="global_local_error_eo_tab"
                        ) as global_local_error_eo_tab:
                            with gr.Row():
                                scatter_equalized_odds = gr.Plot(
                                    label="Equalized Odds", show_label=False
                                )
                                scatter_local_equalized_odds = gr.Plot(
                                    label="Equalized Odds", show_label=False
                                )
                        with gr.Tab(
                            "Treatment Equality", id="global_local_error_te_tab"
                        ) as global_local_error_te_tab:
                            with gr.Row():
                                scatter_treatment_equality = gr.Plot(
                                    label="Treatment Equality", show_label=False
                                )
                                scatter_local_treatment_equality = gr.Plot(
                                    label="Treatment Equality", show_label=False
                                )
                        with gr.Tab(
                            "Consistency", id="global_local_error_const_tab"
                        ) as global_local_error_const_tab:
                            with gr.Row():
                                scatter_consistency = gr.Plot(
                                    label="Consistency", show_label=False
                                )
                                scatter_local_consistency = gr.Plot(
                                    label="Consistency", show_label=False
                                )

                    global_local_error_scatter_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                    ]
                    global_local_error_scatter_output = [
                        scatter_demographic_parity,
                        scatter_local_demographic_parity,
                        scatter_equalized_odds,
                        scatter_local_equalized_odds,
                        scatter_treatment_equality,
                        scatter_local_treatment_equality,
                        scatter_consistency,
                        scatter_local_consistency,
                    ]

                    for tab in [
                        global_local_error_dp_tab,
                        global_local_error_eo_tab,
                        global_local_error_te_tab,
                        global_local_error_const_tab,
                    ]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        ).then(
                            global_local_error_scatter,
                            global_local_error_scatter_input,
                            global_local_error_scatter_output,
                        )

                    start_button_global_local_error_scatter.click(
                        global_local_error_scatter,
                        global_local_error_scatter_input,
                        global_local_error_scatter_output,
                    )

                with gr.Tab("Bias Metrics Correlation"):
                    with gr.Column():
                        with gr.Row():
                            start_button_metric_spm = gr.Button(
                                "Show/Refresh", variant="primary"
                            )
                        with gr.Row():
                            with gr.Tab(
                                "Global Bias Metrics Correlation", id="global_spm_tab"
                            ) as global_spm_tab:
                                with gr.Row():
                                    global_spm = gr.Plot(
                                        label="Bias Metrics Correlation",
                                        show_label=False,
                                    )
                            with gr.Tab(
                                "Local Bias Metrics Correlation", id="local_spm_tab"
                            ) as local_spm_tab:
                                with gr.Row():
                                    local_spm = gr.Plot(
                                        label="Bias Metrics Correlation",
                                        show_label=False,
                                    )

                    metric_spm_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                    ]

                    metric_spm_output = [global_spm, local_spm]

                    start_button_metric_spm.click(
                        metric_spm,
                        metric_spm_input,
                        metric_spm_output,
                    )

                    for tab in [global_spm_tab, local_spm_tab]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        ).then(
                            metric_spm,
                            metric_spm_input,
                            metric_spm_output,
                        )

                with gr.Tab("Tuning Results"):
                    with gr.Column():
                        with gr.Row():
                            start_button_tuning = gr.Button(
                                "Show/Refresh", variant="primary"
                            )
                        (
                            tuning_dp_tab,
                            tuning_dp_global_plot,
                            tuning_dp_local_plot,
                            tuning_dp_error_plot,
                        ) = create_fairness_metric_tab(
                            "Demographic Parity", "tuning_results_dp_tab"
                        )
                        (
                            tuning_eo_tab,
                            tuning_eo_global_plot,
                            tuning_eo_local_plot,
                            tuning_eo_error_plot,
                        ) = create_fairness_metric_tab(
                            "Equalized Odds", "tuning_results_eo_tab"
                        )
                        (
                            tuning_te_tab,
                            tuning_te_global_plot,
                            tuning_te_local_plot,
                            tuning_te_error_plot,
                        ) = create_fairness_metric_tab(
                            "Treatment Equality", "tuning_results_te_tab"
                        )
                        (
                            tuning_const_tab,
                            tuning_const_global_plot,
                            tuning_const_local_plot,
                            tuning_const_error_plot,
                        ) = create_fairness_metric_tab(
                            "Consistency", "tuning_results_const_tab"
                        )


                    tuning_result_plots_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                    ]
                    tuning_result_plots_output = [
                        tuning_dp_global_plot,
                        tuning_dp_local_plot,
                        tuning_dp_error_plot,
                        tuning_eo_global_plot,
                        tuning_eo_local_plot,
                        tuning_eo_error_plot,
                        tuning_te_global_plot,
                        tuning_te_local_plot,
                        tuning_te_error_plot,
                        tuning_const_global_plot,
                        tuning_const_local_plot,
                        tuning_const_error_plot,
                    ]

                    for tab in [tuning_dp_tab, tuning_eo_tab, tuning_te_tab, tuning_const_tab]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        ).then(
                            tuning_result_plots,
                            tuning_result_plots_input,
                            tuning_result_plots_output,
                        )

                    start_button_tuning.click(
                        tuning_result_plots,
                        tuning_result_plots_input,
                        tuning_result_plots_output,
                    )

    with gr.Tab("Run Tests"):
        gr.Markdown(
            """
            Select datasets, algorithm and target metric to start a comparison.
            """
        )
        with gr.Column():
            with gr.Column() as fairness_input_panel:
                selected_datasets = gr.CheckboxGroup(DATA_DICT.keys(), label="Datasets")
                with gr.Row():
                    with gr.Column():
                        file_output = gr.File()
                        upload_button = gr.UploadButton("Click to Upload a File")
                        upload_button.upload(upload_file, upload_button, file_output)
                    with gr.Column():
                        prot_attr = gr.Textbox(label="Protected attribute name")
                        label = gr.Textbox(label="Label name")
                        dataset_button = gr.Button("Insert dataset")
                        dataset_button.click(insert_dataset, [file_output, prot_attr, label], [])
                selected_models = gr.CheckboxGroup(
                    MODEL_NAMES_FOR_EXPERIMENTS, value=[], label="Models", visible=True
                )

                with gr.Row():
                    with gr.Column():
                        selected_fairness_metrics = gr.CheckboxGroup(
                            FAIRNESS_METRICS,
                            value=[
                                "demographic_parity",
                                "equalized_odds",
                                "treatment_equality",
                                "consistency"
                            ],
                            label="Target Fairness Metric",
                            interactive=True,
                            visible=True,
                        )
                    with gr.Column():
                        alt_lambda_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            label="Weight of the fairness term",
                        )
                        alt_local_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0,
                            label="Weight of local fairness",
                        )
                with gr.Row():
                    with gr.Column():
                        test_split_ratio_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.3,
                            step=0.1,
                            label="Test split size",
                        )
                        num_iterations_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            label="Number of iterations",
                        )
                    with gr.Column():
                        alt_time_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0,
                            label="Importance of runtime",
                        )
                        alt_mem_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.8,
                            label="Maximum usage of available memory (in %)",
                        )
                
            is_tuning_enabled = gr.Checkbox(label="Tune hyperparameters?")
            start_algorithms_button = gr.Button(
                "Start Algorithms", variant="primary"
            )
            loading_bar = gr.CheckboxGroup(label="")
            with gr.Column(visible=False) as run_algorithms_output_panel:
                start_algorithms_button.click(
                    run_fairness_algorithms,
                    [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                        test_split_ratio_slider,
                        num_iterations_slider,
                        alt_lambda_slider,
                        alt_local_slider,
                    ],
                    [loading_bar],
                ).then(
                    lambda x: gr.Column(visible=True),
                    inputs=[start_algorithms_button],
                    outputs=[run_algorithms_output_panel],
                )
                with gr.Tab("Results"):
                    gr.Markdown(
                        "### Your results are ready!\n\n<small>Select a tab to see the results.</small>"
                    )
                with gr.Tab("Table") as result_table_tab:
                    result_table = gr.DataFrame()

                    def draw_result_table(datasets, models, result_path):
                        table = pd.read_csv(result_path)
                        table = table.loc[table["dataset"].isin(datasets)]
                        table = table.loc[table["model"].isin(models)]
                        return table

                    result_table_tab.select(
                        draw_result_table,
                        inputs=[
                            selected_datasets,
                            selected_models,
                            gr.Textbox(
                                value="Results/complete_results.csv", visible=False
                            ),
                        ],
                        outputs=[result_table],
                    )

                with gr.Tab("Dataset-specific Recommendations", render=False):
                    with gr.Column():
                        with gr.Tab("Demographic Parity"):
                            with gr.Column():
                                score_demographic_parity = gr.BarPlot(
                                    label="Demographic Parity",
                                    title="Recommendation Score",
                                )
                                table_demographic_parity = gr.Dataframe(type="pandas")
                        with gr.Tab("Equalized Odds"):
                            with gr.Column():
                                score_equalized_odds = gr.BarPlot(
                                    label="Equalized Odds", title="Recommendation Score"
                                )
                                table_equalized_odds = gr.Dataframe(type="pandas")
                        with gr.Tab("Treatment Equality"):
                            with gr.Column():
                                score_treatment_equality = gr.BarPlot(
                                    label="Treatment Equality",
                                    title="Recommendation Score",
                                )
                                table_treatment_equality = gr.Dataframe(type="pandas")
                        with gr.Tab("Consistency"):
                            with gr.Column():
                                score_consistency = gr.BarPlot(
                                    label="Consistency",
                                    title="Recommendation Score",
                                )
                                table_consistency = gr.Dataframe(type="pandas")
                        with gr.Column():
                            # alt_lambda_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Weight of the fairness term")
                            # alt_local_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0, label="Weight of local fairness")
                            # datasize_slider = gr.Slider(minimum=1000, maximum=100000, step=1000, value=10000, label="Expected size of the dataset")
                            # alt_time_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0, label="Importance of runtime")
                            # alt_mem_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.8, label="Maximum usage of available memory (in %)")
                            start_button_recommendation = gr.Button(
                                "Show/Refresh", variant="primary"
                            )

                with gr.Tab("Experimental Results") as experimental_results_tab:
                    with gr.Column():
                        (
                            exp_dp_tab,
                            exp_dp_global_plot,
                            exp_dp_local_plot,
                            exp_dp_error_plot,
                        ) = create_fairness_metric_tab(
                            "Demographic Parity", "experimental_results_dp_tab"
                        )
                        (
                            exp_eo_tab,
                            exp_eo_global_plot,
                            exp_eo_local_plot,
                            exp_eo_error_plot,
                        ) = create_fairness_metric_tab(
                            "Equalized Odds", "experimental_results_eo_tab"
                        )
                        (
                            exp_te_tab,
                            exp_te_global_plot,
                            exp_te_local_plot,
                            exp_te_error_plot,
                        ) = create_fairness_metric_tab(
                            "Treatment Equality", "experimental_results_te_tab"
                        )
                        (
                            exp_const_tab,
                            exp_const_global_plot,
                            exp_const_local_plot,
                            exp_const_error_plot,
                        ) = create_fairness_metric_tab(
                            "Consistency", "experimental_results_const_tab"
                        )

                    generate_general_recommendation_plots_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                        gr.Textbox(value="Results/complete_results.csv", visible=False),
                    ]
                    generate_general_recommendation_plots_output = [
                        exp_dp_global_plot,
                        exp_dp_local_plot,
                        exp_dp_error_plot,
                        exp_eo_global_plot,
                        exp_eo_local_plot,
                        exp_eo_error_plot,
                        exp_te_global_plot,
                        exp_te_local_plot,
                        exp_te_error_plot,
                        exp_const_global_plot,
                        exp_const_local_plot,
                        exp_const_error_plot,
                    ]
                    experimental_results_tab.select(
                        generate_general_fairness_plots,
                        generate_general_recommendation_plots_input,
                        generate_general_recommendation_plots_output,
                    )
                    for tab in [exp_dp_tab, exp_eo_tab, exp_te_tab, exp_const_tab]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        )
                        tab.select(
                            generate_general_fairness_plots,
                            generate_general_recommendation_plots_input,
                            generate_general_recommendation_plots_output,
                        )

                with gr.Tab(
                    "Global/Local and Global/Error Correlation"
                ) as global_local_error_tab:
                    with gr.Column():
                        with gr.Tab(
                            "Demographic Parity", id="global_local_error_dp_tab"
                        ) as global_local_error_dp_tab:
                            with gr.Row():
                                scatter_demographic_parity = gr.Plot(
                                    label="Demographic Parity", show_label=False
                                )
                                scatter_local_demographic_parity = gr.Plot(
                                    label="Demographic Parity", show_label=False
                                )
                        with gr.Tab(
                            "Equalized Odds", id="global_local_error_eo_tab"
                        ) as global_local_error_eo_tab:
                            with gr.Row():
                                scatter_equalized_odds = gr.Plot(
                                    label="Equalized Odds", show_label=False
                                )
                                scatter_local_equalized_odds = gr.Plot(
                                    label="Equalized Odds", show_label=False
                                )
                        with gr.Tab(
                            "Treatment Equality", id="global_local_error_te_tab"
                        ) as global_local_error_te_tab:
                            with gr.Row():
                                scatter_treatment_equality = gr.Plot(
                                    label="Treatment Equality", show_label=False
                                )
                                scatter_local_treatment_equality = gr.Plot(
                                    label="Treatment Equality", show_label=False
                                )
                        with gr.Tab(
                            "Consistency", id="global_local_error_const_tab"
                        ) as global_local_error_const_tab:
                            with gr.Row():
                                scatter_consistency = gr.Plot(
                                    label="Consistency", show_label=False
                                )
                                scatter_local_consistency = gr.Plot(
                                    label="Consistency", show_label=False
                                )

                    global_local_error_scatter_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                        gr.Textbox(value="Results/complete_results.csv", visible=False),
                    ]
                    global_local_error_scatter_output = [
                        scatter_demographic_parity,
                        scatter_local_demographic_parity,
                        scatter_equalized_odds,
                        scatter_local_equalized_odds,
                        scatter_treatment_equality,
                        scatter_local_treatment_equality,
                        scatter_consistency,
                        scatter_local_consistency,
                    ]

                    global_local_error_tab.select(
                        global_local_error_scatter,
                        global_local_error_scatter_input,
                        global_local_error_scatter_output,
                    )
                    for tab in [
                        global_local_error_dp_tab,
                        global_local_error_eo_tab,
                        global_local_error_te_tab,
                        global_local_error_const_tab,
                    ]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        )
                        tab.select(
                            global_local_error_scatter,
                            global_local_error_scatter_input,
                            global_local_error_scatter_output,
                        )

                    start_button_global_local_error_scatter.click(
                        global_local_error_scatter,
                        global_local_error_scatter_input,
                        global_local_error_scatter_output,
                    )

                with gr.Tab("Bias Metrics Correlation") as bias_metrics_tab:
                    with gr.Column():
                        with gr.Row():
                            with gr.Tab(
                                "Global Bias Metrics Correlation", id="global_spm_tab"
                            ) as global_spm_tab:
                                with gr.Row():
                                    global_spm = gr.Plot(
                                        label="Bias Metrics Correlation",
                                        show_label=False,
                                    )
                            with gr.Tab(
                                "Local Bias Metrics Correlation", id="local_spm_tab"
                            ) as local_spm_tab:
                                with gr.Row():
                                    local_spm = gr.Plot(
                                        label="Bias Metrics Correlation",
                                        show_label=False,
                                    )

                    metric_spm_input = [
                        selected_datasets,
                        selected_models,
                        selected_fairness_metrics,
                        is_tuning_enabled,
                        gr.Textbox(value="Results/complete_results.csv", visible=False),
                    ]

                    metric_spm_output = [global_spm, local_spm]

                    start_button_metric_spm.click(
                        metric_spm,
                        metric_spm_input,
                        metric_spm_output,
                    )
                    bias_metrics_tab.select(
                        metric_spm,
                        metric_spm_input,
                        metric_spm_output,
                    )
                    for tab in [global_spm_tab, local_spm_tab]:
                        tab.select(
                            refresh_plots,
                            inputs=[],
                            outputs=tab.children[0].children,
                        )
                        tab.select(
                            metric_spm,
                            metric_spm_input,
                            metric_spm_output,
                        )

                with gr.Tab("Tuning Results", render=False):
                    with gr.Column():
                        with gr.Row():
                            start_button_tuning = gr.Button(
                                "Show/Refresh", variant="primary"
                            )
                        (
                            tuning_dp_tab,
                            tuning_dp_global_plot,
                            tuning_dp_local_plot,
                            tuning_dp_error_plot,
                        ) = create_fairness_metric_tab(
                            "Demographic Parity", "tuning_results_dp_tab"
                        )
                        (
                            tuning_eo_tab,
                            tuning_eo_global_plot,
                            tuning_eo_local_plot,
                            tuning_eo_error_plot,
                        ) = create_fairness_metric_tab(
                            "Equalized Odds", "tuning_results_eo_tab"
                        )
                        (
                            tuning_te_tab,
                            tuning_te_global_plot,
                            tuning_te_local_plot,
                            tuning_te_error_plot,
                        ) = create_fairness_metric_tab(
                            "Treatment Equality", "tuning_results_te_tab"
                        )


if __name__ == "__main__":
    fairness_dashboard.launch()
