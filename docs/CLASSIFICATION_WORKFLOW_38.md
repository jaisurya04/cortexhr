# Complete End-to-End Data Science Workflow for Classification

## 1. Business Understanding

1. Understand the business problem and why it matters.
2. Identify the business objective and expected outcome.
3. Identify the stakeholders and end users.
4. Understand how the current process works.
5. Identify the limitations of the current process.
6. Define the business questions the model should answer.
7. Identify the cost of false positives and false negatives.
8. Define measurable business success criteria.
9. Identify operational, ethical, legal, and financial constraints.
10. Confirm whether machine learning is actually necessary.

## 2. Problem Definition

1. Convert the business problem into a classification problem.
2. Identify the target variable.
3. Define each target class clearly.
4. Identify whether the problem is binary or multiclass classification.
5. Define the prediction unit, such as one customer or one transaction.
6. Define when the prediction will be made.
7. Define the prediction horizon if applicable.
8. Define which information is available at prediction time.
9. Define the expected model output.
10. Define how the prediction will be used in the business process.

## 3. Define Evaluation Criteria

1. Choose a primary evaluation metric based on the business objective.
2. Choose secondary evaluation metrics.
3. Define the minimum acceptable performance.
4. Define the baseline performance that the model must beat.
5. Define the cost of different classification errors.
6. Decide whether ranking, probability estimation, or hard classification is most important.
7. Define model latency and throughput requirements if needed.
8. Define fairness, explainability, and reliability requirements.
9. Define the final acceptance criteria before model development.

## 4. Data Collection

1. Identify all available data sources.
2. Collect the relevant datasets.
3. Understand the meaning of each dataset and its columns.
4. Identify the data owner and collection method.
5. Understand the time period covered by the data.
6. Check data access permissions and privacy requirements.
7. Verify that the data is relevant to the business problem.
8. Confirm that the target labels are available and meaningful.
9. Record data provenance and collection limitations.
10. Save an untouched copy of the original data.

## 5. Data Loading and Initial Inspection

1. Load the dataset using appropriate tools.
2. Inspect the number of rows and columns.
3. Display sample records.
4. Inspect column names and data types.
5. Identify the target column.
6. Check duplicate column names.
7. Check duplicate records.
8. Inspect unique values in categorical columns.
9. Inspect numerical summaries.
10. Verify whether the dataset matches the expected schema.

## 6. Data Understanding and Data Dictionary

1. Identify the meaning of every feature.
2. Classify columns into numerical, categorical, datetime, text, identifier, and target types.
3. Identify unique identifiers and potential join keys.
4. Identify features available at prediction time.
5. Identify features created after the target event.
6. Document valid ranges and expected values.
7. Identify business rules and domain constraints.
8. Create a data dictionary.
9. Record assumptions and unresolved questions.
10. Confirm the target definition with the data owner.

## 7. Data Quality Audit

1. Check missing values in every column.
2. Calculate missing-value percentages.
3. Identify duplicate rows and duplicate entities.
4. Check incorrect data types.
5. Detect invalid numerical values.
6. Check impossible dates and inconsistent date formats.
7. Detect inconsistent categorical labels.
8. Check inconsistent units and measurement scales.
9. Detect suspicious extreme values.
10. Check contradictory values between related columns.
11. Identify unexpected target labels.
12. Check class distribution and target-label quality.
13. Identify data collection errors and structural missingness.
14. Document every issue and its proposed resolution.
15. Fix confirmed errors without destroying valid information.

## 8. Exploratory Data Analysis (EDA)

1. Examine the distribution of the target variable.
2. Calculate class counts and class percentages.
3. Analyze numerical feature distributions.
4. Analyze categorical feature frequencies.
5. Use histograms and density plots where useful.
6. Use box plots to inspect distributions and potential outliers.
7. Analyze feature relationships using appropriate visualizations.
8. Compare feature distributions across target classes.
9. Examine numerical correlations and possible multicollinearity.
10. Investigate time-based patterns if timestamps are available.
11. Investigate subgroup differences and potential data bias.
12. Identify suspicious patterns that may indicate leakage.
13. Form hypotheses based on the observations.
14. Validate hypotheses instead of treating correlations as causes.
15. Summarize the important findings and their business implications.

## 9. Data Leakage Investigation

1. Identify features that directly encode the target.
2. Identify features created after the outcome occurred.
3. Check whether any features would be unavailable at prediction time.
4. Inspect suspiciously predictive features.
5. Check whether duplicate entities appear across training and test sets.
6. Check whether preprocessing uses information from validation or test data.
7. Check whether target-derived aggregates are calculated incorrectly.
8. Check whether temporal information crosses the prediction cutoff.
9. Remove or redesign invalid features.
10. Document the leakage checks and decisions.

## 10. Data Cleaning

1. Remove confirmed duplicate records according to the data definition.
2. Correct invalid data types.
3. Standardize categorical values.
4. Fix inconsistent units and formats.
5. Handle impossible values according to domain rules.
6. Investigate outliers before modifying them.
7. Preserve legitimate rare observations.
8. Handle missing values using an appropriate strategy.
9. Resolve conflicting records where possible.
10. Validate the cleaned dataset against business rules.
11. Record all cleaning decisions.
12. Keep the original data unchanged for traceability.

## 11. Feature Engineering

1. Identify useful features based on domain knowledge.
2. Create numerical ratios and differences where meaningful.
3. Extract useful date and time features.
4. Create frequency, recency, and aggregation features where appropriate.
5. Group rare categories when justified.
6. Create interaction features when they have a clear rationale.
7. Encode meaningful business concepts as features.
8. Avoid features that use future information.
9. Ensure aggregated features respect entity and time boundaries.
10. Document the purpose of every engineered feature.
11. Validate that engineered features are available at inference time.
12. Compare feature sets using validation performance.

## 12. Train, Validation, and Test Split

1. Define the independent unit of analysis.
2. Choose a random, stratified, grouped, or temporal split based on the data structure.
3. Separate training, validation, and test data when the workflow requires them.
4. Preserve the target-class distribution where appropriate.
5. Keep related records together when entity leakage is possible.
6. Use chronological splitting for future prediction problems when necessary.
7. Prevent duplicate entities from leaking across splits.
8. Keep the final test set untouched during model selection.
9. Record the split proportions and random seed.
10. Verify that the split reflects the intended real-world use case.

## 13. Build a Preprocessing Pipeline

1. Separate features from the target variable.
2. Identify numerical and categorical preprocessing requirements.
3. Impute missing numerical values.
4. Impute missing categorical values.
5. Scale numerical features when required by the model.
6. Encode categorical features appropriately.
7. Handle unknown categories at inference time.
8. Apply text preprocessing if text features are used.
9. Apply feature selection only within the training workflow.
10. Fit all learned preprocessing steps on training data only.
11. Integrate preprocessing with the model using a reusable pipeline.
12. Apply the same fitted pipeline during validation, testing, and production inference.
13. Verify that the pipeline handles unseen and missing input values safely.

## 14. Establish Baseline Models

1. Build a simple rule-based baseline if appropriate.
2. Build a majority-class or dummy classifier.
3. Establish a simple, interpretable machine learning baseline.
4. Evaluate the baseline using the chosen metrics.
5. Record the confusion matrix.
6. Compare the results with the business requirements.
7. Identify the weaknesses of the baseline.
8. Use the baseline as a reference for future improvements.

## 15. Model Selection

1. Select candidate algorithms suitable for the problem.
2. Consider Logistic Regression as an interpretable baseline.
3. Consider Decision Trees for rule-based relationships.
4. Consider Random Forest for nonlinear patterns.
5. Consider gradient-boosting models for tabular data.
6. Consider Support Vector Machines when suitable.
7. Consider Naive Bayes for appropriate feature distributions.
8. Consider neural networks when the data and problem justify them.
9. Consider model complexity, training cost, inference speed, and explainability.
10. Select a manageable set of candidate models.

## 16. Train Candidate Models

1. Fit each model using training data.
2. Keep learned preprocessing inside the training pipeline.
3. Use class weights when justified by the business objective.
4. Consider resampling methods only when appropriate.
5. Apply oversampling or undersampling inside training folds only.
6. Record model parameters and training settings.
7. Track training time and computational requirements.
8. Save reproducible training configurations.
9. Check for training failures and convergence warnings.
10. Compare training performance with validation performance.

## 17. Cross-Validation

1. Choose an appropriate cross-validation strategy.
2. Use stratified folds for suitable classification datasets.
3. Use grouped folds when records from the same entity are dependent.
4. Use time-aware validation for temporal prediction problems.
5. Fit preprocessing separately inside every training fold.
6. Evaluate the model on each held-out fold.
7. Calculate the mean and variation of the evaluation metrics.
8. Compare candidate models on consistent folds.
9. Check for overfitting and unstable results.
10. Avoid using the final test set for cross-validation or model selection.

## 18. Handle Class Imbalance

1. Inspect the target-class distribution.
2. Determine whether imbalance affects the business objective.
3. Avoid relying on accuracy alone.
4. Consider class-weighted learning.
5. Consider random oversampling or undersampling when appropriate.
6. Consider SMOTE only when its assumptions are suitable.
7. Apply resampling only to training data or training folds.
8. Compare methods using precision, recall, PR-AUC, and business costs.
9. Inspect the effect on probability calibration.
10. Choose the approach based on validation results rather than assumptions.

## 19. Hyperparameter Tuning

1. Identify the most important hyperparameters.
2. Define sensible search ranges.
3. Choose grid search, random search, or Bayesian optimization as appropriate.
4. Tune models using training data and cross-validation.
5. Optimize the predefined primary metric.
6. Track computational cost.
7. Compare tuned models against their untuned baselines.
8. Check whether improvements are consistent across folds.
9. Avoid overfitting the validation process through excessive experimentation.
10. Record the final hyperparameters and selection rationale.

## 20. Model Evaluation

1. Evaluate the selected model on validation data.
2. Generate predictions and predicted probabilities where supported.
3. Calculate the confusion matrix.
4. Calculate accuracy.
5. Calculate precision.
6. Calculate recall (sensitivity).
7. Calculate specificity where relevant.
8. Calculate F1-score.
9. Calculate ROC-AUC.
10. Calculate PR-AUC when positive-class detection is important.
11. Inspect the classification report.
12. Examine the precision-recall curve.
13. Examine the ROC curve.
14. Compare performance with the baseline and business requirements.
15. Evaluate performance across relevant subgroups.
16. Estimate uncertainty where appropriate.
17. Record limitations and possible failure modes.

## 21. Threshold Selection

1. Understand the default classification threshold.
2. Identify the relative costs of false positives and false negatives.
3. Examine precision and recall at different thresholds.
4. Select a threshold using validation data only.
5. Optimize the metric or business cost defined in advance.
6. Check the operational volume of predicted positive cases.
7. Confirm that the threshold is feasible for the review team.
8. Document the reason for the chosen threshold.
9. Freeze the threshold before final test evaluation.
10. Revisit the threshold only through a controlled evaluation process.

## 22. Probability Calibration

1. Determine whether predicted probabilities need to represent actual event likelihoods.
2. Compare predicted probabilities with observed event frequencies.
3. Inspect a calibration curve.
4. Calculate the Brier score or another suitable probability metric.
5. Consider Platt scaling or isotonic regression where appropriate.
6. Fit the calibrator using training or validation predictions without using final test labels.
7. Preserve separation between calibration data and final test data.
8. Recheck calibration on held-out data.
9. Explain the difference between ranking scores and calibrated probabilities.
10. Avoid presenting uncalibrated scores as reliable probabilities.

## 23. Feature Importance and Model Explainability

1. Inspect model coefficients when applicable.
2. Examine feature importance for tree-based models.
3. Use permutation importance where suitable.
4. Use SHAP or other explanation methods when appropriate.
5. Identify the strongest predictive features.
6. Check whether explanations make business sense.
7. Investigate unexpected or suspiciously powerful features.
8. Distinguish correlation from causation.
9. Explain individual predictions when required.
10. Communicate uncertainty and explanation limitations.
11. Ensure explanations do not encourage unfair or inappropriate decisions.

## 24. Error Analysis

1. Inspect false positives.
2. Inspect false negatives.
3. Examine true positives and true negatives.
4. Identify common error patterns.
5. Compare errors across customer, product, geographic, or other relevant groups.
6. Investigate data-quality problems behind errors.
7. Check whether errors are concentrated in rare cases.
8. Determine whether errors are caused by poor features, labels, or model limitations.
9. Prioritize improvements based on business impact.
10. Document unresolved error patterns.

## 25. Fairness, Privacy, and Responsible AI

1. Identify relevant protected and vulnerable groups.
2. Check whether the data reflects historical discrimination.
3. Evaluate appropriate subgroup performance metrics.
4. Examine differences in false-positive and false-negative rates.
5. Investigate potential proxy variables.
6. Apply relevant legal and organizational requirements.
7. Minimize the use of sensitive personal data.
8. Restrict access to confidential information.
9. Document limitations and potential harms.
10. Establish human oversight for consequential decisions.
11. Ensure the model is not used beyond its validated purpose.

## 26. Robustness and Stability Testing

1. Test the model on different validation samples where appropriate.
2. Check sensitivity to reasonable preprocessing changes.
3. Examine performance across relevant subgroups.
4. Test unusual but valid input values.
5. Test missing and unseen categories.
6. Check sensitivity to class-distribution changes.
7. Evaluate temporal stability when historical data is available.
8. Test performance under plausible real-world changes.
9. Identify conditions where the model becomes unreliable.
10. Define fallback behavior for invalid or out-of-distribution inputs.

## 27. Final Test Evaluation

1. Freeze the model and preprocessing pipeline.
2. Freeze hyperparameters and the decision threshold.
3. Freeze any probability calibrator.
4. Evaluate the complete pipeline on the untouched test set.
5. Calculate all predefined final evaluation metrics.
6. Compare test performance with validation performance.
7. Investigate unexpected performance differences without tuning to the test set.
8. Report uncertainty and relevant subgroup results.
9. Document the final model's limitations.
10. Decide whether the model meets the predefined acceptance criteria.

## 28. Final Model Selection

1. Compare candidate models against the baseline.
2. Review predictive performance and generalization.
3. Consider interpretability and maintainability.
4. Consider inference latency and infrastructure cost.
5. Consider fairness, privacy, and operational risks.
6. Choose the model that best satisfies the business requirements.
7. Record the selection rationale.
8. Save the final model and fitted preprocessing pipeline.
9. Record model version, training data version, and configuration.
10. Avoid selecting a model solely because one metric is highest.

## 29. Build a Reusable Prediction Pipeline

1. Package preprocessing and prediction into one workflow.
2. Validate incoming feature names and data types.
3. Validate required input columns.
4. Handle missing values and unknown categories safely.
5. Generate predicted classes or scores as required.
6. Apply the approved threshold or risk-band rules.
7. Return consistent, documented outputs.
8. Add informative error handling.
9. Test the pipeline on representative examples.
10. Verify reproducibility between development and production environments.

## 30. Testing and Quality Assurance

1. Write unit tests for data transformations.
2. Write tests for feature engineering.
3. Test missing-value handling.
4. Test categorical encoding and unseen categories.
5. Test prediction output types and shapes.
6. Test invalid input rejection.
7. Test model loading and inference.
8. Test the API or application integration.
9. Test edge cases and representative examples.
10. Run regression tests when the pipeline changes.
11. Check for data leakage in the complete workflow.
12. Document test results and unresolved issues.

## 31. Model Packaging and Reproducibility

1. Save the trained model using an appropriate format.
2. Save preprocessing components and calibrators.
3. Record model dependencies and software versions.
4. Store training parameters and random seeds.
5. Track data and model versions.
6. Record evaluation metrics and experiment results.
7. Document how to load and use the model.
8. Avoid storing secrets or unnecessary sensitive data.
9. Create a reproducible training procedure.
10. Verify that the saved model reproduces expected predictions.

## 32. Deployment Planning

1. Decide whether batch, real-time, or streaming inference is required.
2. Choose a suitable serving architecture.
3. Define API inputs and outputs if an API is needed.
4. Define authentication and authorization requirements.
5. Define expected latency, throughput, and availability.
6. Estimate infrastructure and maintenance costs.
7. Define logging, monitoring, and alerting requirements.
8. Establish a rollback strategy.
9. Define human review and escalation procedures.
10. Prepare deployment documentation.

## 33. Deployment

1. Package the model and its dependencies.
2. Deploy to a development or staging environment.
3. Verify the deployed model loads correctly.
4. Run integration and smoke tests.
5. Compare deployed predictions with local predictions.
6. Validate input and output contracts.
7. Check security and access controls.
8. Deploy to production only after approval.
9. Monitor the initial production rollout.
10. Maintain a rollback option if problems occur.

## 34. Monitoring

1. Monitor service availability and prediction latency.
2. Monitor errors and failed requests.
3. Monitor input data quality.
4. Monitor missing-value and unknown-category rates.
5. Monitor feature distribution changes.
6. Monitor prediction and class distributions.
7. Monitor data drift and concept drift where measurable.
8. Monitor model performance when ground-truth labels become available.
9. Monitor subgroup performance and fairness.
10. Alert the team when predefined limits are exceeded.
11. Record incidents and corrective actions.

## 35. Model Maintenance and Retraining

1. Define when retraining is necessary.
2. Collect new data with reliable labels.
3. Validate new data before training.
4. Re-run data-quality and leakage checks.
5. Re-train using the approved pipeline.
6. Compare the new model with the existing production model.
7. Evaluate the new model on appropriate held-out data.
8. Review calibration, fairness, and robustness.
9. Approve the new model before deployment.
10. Version the model and retain rollback capability.
11. Track whether retraining improves real-world outcomes.

## 36. Documentation and Reporting

1. Document the business problem.
2. Document the target definition and prediction timing.
3. Document the data sources and data dictionary.
4. Document data-quality issues and cleaning decisions.
5. Document feature engineering and preprocessing.
6. Document the train-validation-test strategy.
7. Document baseline and candidate models.
8. Document evaluation metrics and results.
9. Document threshold and calibration decisions.
10. Document limitations, risks, and fairness checks.
11. Document deployment and monitoring procedures.
12. Provide instructions for reproducing the project.

## 37. Final Project Review

1. Confirm that all business requirements are addressed.
2. Confirm that the model beats the required baseline.
3. Confirm that the test set remained untouched during model selection.
4. Confirm that no known leakage remains.
5. Confirm that the complete pipeline works end to end.
6. Confirm that tests pass.
7. Confirm that deployment and monitoring requirements are met.
8. Confirm that documentation is complete.
9. Present the results, limitations, and recommendations.
10. Identify future improvements based on evidence.

## 38. Continuous Improvement

1. Collect feedback from actual users.
2. Compare model predictions with real-world outcomes.
3. Investigate recurring errors and operational problems.
4. Improve data collection where necessary.
5. Add or revise features only when justified.
6. Reassess business metrics and acceptance criteria.
7. Review fairness and privacy risks periodically.
8. Test improvements using a controlled evaluation process.
9. Update documentation and version history.
10. Retire the model when it no longer meets business needs.
