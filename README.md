# FairCR – an evaluation and recommendation system for fair classification algorithms

This repository contains the code of our ICDE 2024 demo paper. Thanks to Linus Zaiser (University of Stuttgart) for helping to update the system.<br />

This project uses Angular for frontend development and Flask for backend development.\
This repository includes a script to automatically start both the Angular development server and the Flask server.

## Prerequisites

> - Atleast Python 3.8 must be installed
> - Node.js (recommended LTS Version): [nodejs.org](https://nodejs.org/en)
> - Angular CLI: Install it globally using npm: 
 
```bash
npm install -g @angular/cli
```

## Initial Configuration
To install all dependencies run: 
> ### Linux & Mac 
> ```shell
> sh scripts/initialize.sh
> ```
> 
> ### Windows
> ```shell
> .\scripts\initialize.bat
> ```

##  Run
To start the application (WHILE DEVELOPING) run:  \
This will start the Backend API Server in an extra terminal \
Open the application : http://localhost:4200/
> ### Linux & Mac 
> ```shell
> sh scripts/run.sh
> ```
> 
> ### Windows
> ```shell
> .\scripts\run.bat
> ```


## General Information

- Each component comes with an information field. Click it to get more details of the component. <br />
- While experiments with the individual fairness metric "consistency" and global fairness metric "equal_opportunity" can be run, the initially provided evaluation results do not contain them. Thus, you first have to run tests, before being being reflected in the general recommendation tab.<br />
- The memory usage and runtime (for the general recommendation) is predicted based on the results of preliminary experiments.<br />
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
- Add the name of the algorithm in `angular-project/src/app/run-analyzer/run-input-field/run-input-field.component.html`.<br />


## Add New Metrics

- For the parameter optimization component, add the metric to `algorithm/evaluation/eval_classifier.py`. The validation dataset, predictions and sensitive attributes are given as input. Thus, these informations are enough to build own fairness metrics.<br />
- Similarily, the metric has to be added to `evaluation.py`.<br />
- Add the name of the metric to the options in `angular-project/src/app/run-analyzer/run-input-field/run-input-field.component.html`.<br />
