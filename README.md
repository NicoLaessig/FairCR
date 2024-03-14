# FairCR – an evaluation and recommendation system for fair classification algorithms

This repository contains the code of our ICDE 2024 demo paper.

## General Information

- You need to run `demo.py` to start the demo.<br />
- While experiments with the individual fairness metric "consistency" can be run, the initially provided evaluation results do not contain them. Thus, you first have to run tests, before being able to recommend algorithms on the given datasets. For the group fairness metrics, several iterations have been run for each algorithm.<br />
- You need to restart the app after inserting a dataset.<br />
- The memory usage and runtime is predicted based on the results of preliminary experiments.<br />
- Uploading datasets does not work on every OS. Some plots (on new datasets) might require tests on multiple metrics (for the relation scatterplots).<br />
- For the recommendation system, we use existing fair classification approaches and, thus, do reuse some existing methods. Please check out the respective papers and GitHubs of these methods.<br />
- For some approaches, some slight adaptations had to be made to integrate them in our framework, but the general approach was not altered.<br />
