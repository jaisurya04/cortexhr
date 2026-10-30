# CortexHR chatbot model improvement — Phase 2

## What changed

- Preserved the original `03_Data/csv/chatbot_nlu_training.csv` (858 labelled examples, 33 intents).
- Added 264 curated, human-authored paraphrases in `03_Data/csv/chatbot_nlu_augmented_examples.csv`.
- Created `03_Data/csv/chatbot_nlu_training_expanded.csv` (1,122 examples total). Added examples are not exact copies of the supplied 132-row test messages.
- Production API and chatbot workflow notebook now prefer the expanded training file when it exists.
- Compared word-only TF-IDF Logistic Regression, word+character TF-IDF Logistic Regression, and Multinomial Naive Bayes with stratified 5-fold CV.
- Generated per-intent classification report, confusion matrix, test predictions, and misclassification list.

## Current supplied-test diagnostic

| Metric | Result |
|---|---:|
| Training examples | 1,122 |
| Added curated examples | 264 |
| Intent classes | 33 |
| Supplied test examples | 132 |
| Correct | 119 / 132 |
| Incorrect | 13 / 132 |
| Accuracy | 90.15% |
| Macro-F1 | 90.09% |
| Weighted-F1 | 90.09% |

Previous diagnostic was 107/132 (81.06%). The new run improves this to 119/132 (90.15%), a gain of 12 correct examples and 9.09 percentage points.

## Important evaluation limitation

The supplied test set was used as a development diagnostic while investigating error patterns, so this score must **not** be described as a fully untouched estimate of real-world generalization. The test texts were not copied verbatim into the added training data, but iterative development based on test-set errors still creates some evaluation bias. For a defensible final evaluation, collect a new independently labelled test set of natural user messages, including paraphrases, typos, ambiguous intents, and out-of-scope queries.

Do not claim 100% accuracy. Keep the 13 remaining misclassifications in `chatbot_advanced_misclassifications.csv` as explicit next-step work. Evaluate the model using accuracy, macro-F1, per-intent precision/recall/F1, privacy/permission regression tests, and a fresh test set.
