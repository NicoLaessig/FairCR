# FairCR – an evaluation and recommendation system for fair classification algorithms

This repository contains the code of our ICDE 2024 demo paper.

## General Information

- You need to run `demo.py` to start the demo.<br />
- While experiments with the individual fairness metric "consistency" can be run, the initially provided evaluation results do not contain them. Thus, you first have to run tests, before being able to recommend algorithms on the given datasets. For the group fairness metrics, several iterations have been run for each algorithm.<br />
- The memory usage and runtime is predicted based on the results of preliminary experiments.<br />
- For the recommendation system, we use existing fair classification approaches and, thus, do reuse some existing methods. Please check out the respective papers and GitHubs of these methods.<br />
- For some approaches, some slight adaptations had to be made to integrate them in our framework, but the general approach was not altered.<br />


## Adding Datasets

- Can be done through the app (dataset has to be numerical).<br />
- You need to restart the app after inserting a dataset.<br />
- Some plots (on new datasets) might require tests on multiple metrics (for the relation scatterplots).<br />
- Uploading datasets might not work on every OS (the respective function might have to be adapted).<br /> 


## Add Fair Classifier

- The requirements for adding a new fair classifier:<br /> 
  - Classifier has to have a fit() function, where the input is X_train and y_train (in DataFrame format), and a predict() function, where X_predict is the input.<br /> 
  - All other important parameters are given via Class initialization. An additional required parameter is df_dict, which contains important information about the input dataset.<br />
- Add the default parameter values and values for the grid search parameter optimization to `configs/params.json`<br />
- Add the respective new Class to the `algorithm/__init__.py`.<br />
- Add the class call to the big if-else block in `main.py`<br />
- Add the name of the classifier to the MODELS_LIST in `demo.py`<br />


## Add New Metrics

- For the parameter optimization component, add the metric to `algorithm/evaluation/eval_classifier.py`. The validation dataset, predictions and sensitive attributes are given as input. Thus, these informations are enough to build own fairness metrics.<br />
- Similarily, the metric has to be added to `evaluation.py`.<br />
- Add the name of the metric to the FAIRNESS_METRICS in `demo.py`.<br />
- For the visualization plots, some more changes might currently be required in `demo.py`.<br />
