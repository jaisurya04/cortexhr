# Workflow coverage: your 37-point classification workflow -> where it is implemented

Every item is a numbered section of `05_Starter/notebooks/05_complete_classification_workflow.ipynb` (requirement-by-requirement coverage table + real code + outputs). Task notebooks go deeper on their own phase.
The supplied workflow text ended at item 37 (its first bullet); if more items exist they are not covered yet.

| # | Item | Master notebook section | Also in |
|---|---|---|---|
| 1 | Business Understanding | section 1 | 01_data_audit (stages 1-3), 05 §1-3 |
| 2 | Problem Definition | section 2 | 05 only |
| 3 | Problem Feasibility | section 3 | 05 only |
| 4 | Data Collection | section 4 | 01_data_audit |
| 5 | Data Loading and Initial Inspection | section 5 | 01_data_audit |
| 6 | EDA | section 6 | 01_data_audit stage 4 |
| 7 | Missing-Value Analysis | section 7 | 01_data_audit |
| 8 | Duplicate-Record Analysis | section 8 | 01_data_audit |
| 9 | Outlier Analysis | section 9 | 01_data_audit |
| 10 | Data Consistency and Validity | section 10 | 01_data_audit stage 5 |
| 11 | Data Leakage Investigation | section 11 | 01_data_audit stage 3, 02_baselines leaky demo |
| 12 | Target Variable Preparation | section 12 | 05 only |
| 13 | Feature Identification | section 13 | 05 only |
| 14 | Train/Validation/Test Split | section 14 | 05 only |
| 15 | Numerical Preprocessing | section 15 | 01b_feature_pipeline |
| 16 | Categorical Preprocessing | section 16 | 01b_feature_pipeline |
| 17 | Datetime and Text Preprocessing | section 17 | 01b_feature_pipeline (cutoff) |
| 18 | Preprocessing Pipeline | section 18 | 01b_feature_pipeline |
| 19 | Feature Engineering | section 19 | 01b_feature_pipeline |
| 20 | Feature Selection | section 20 | 05 only |
| 21 | Dimensionality Reduction | section 21 | 05 only |
| 22 | Class Imbalance Analysis | section 22 | 02_baselines |
| 23 | Imbalance Handling Techniques | section 23 | 02_baselines (SMOTE) |
| 24 | Establish a Baseline | section 24 | 02_baselines |
| 25 | Candidate Algorithms | section 25 | 02_baselines |
| 26 | Initial Model Training | section 26 | 02_baselines |
| 27 | Cross-Validation | section 27 | 02_baselines |
| 28 | Hyperparameter Tuning | section 28 | 02_baselines stage 10 |
| 29 | Threshold Optimization | section 29 | 03_model_evaluation stage 10 |
| 30 | Confusion Matrix | section 30 | 03_model_evaluation stage 9 |
| 31 | Classification Metrics | section 31 | 03_model_evaluation stage 9 |
| 32 | Metric Selection | section 32 | 05 only |
| 33 | Probability and Calibration | section 33 | 03_model_evaluation (calibration) |
| 34 | Error Analysis | section 34 | 03_model_evaluation stage 11 |
| 35 | Over/Underfitting | section 35 | 02_baselines (tree vs CV) |
| 36 | Model Comparison and Selection | section 36 | 02_baselines |
| 37 | Final Model Training | section 37 | 03_model_evaluation (test scored once) |

Algorithms not runnable here (libraries not installed in the build environment): XGBoost, LightGBM, CatBoost and Bayesian optimisation (optuna). The cells are guarded: they run where the library is installed and say so where it is not.
