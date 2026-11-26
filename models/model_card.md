# Model Card: Final AI Capstone Classifier

## 1. Model Name

LogisticRegression_TFIDF

## 2. Model Type

TF-IDF text feature extraction with a
LogisticRegression
classifier.

## 3. Training Data

- Dataset: data/expanded_dataset.csv
- Total records: 200
- Number of categories: 10
- Features:
  - Combined system prompt and test question
  - Question word count
  - Question-to-prompt length ratio
- Target: category
- Training date: 2026-11-26

## 4. Model Selection Metric

- Metric: Accuracy
- Previously recorded test accuracy: 1.0000
- Previous evaluation: stratified 80/20 train-test split
- Model selection: highest recorded accuracy
- Tie-breaking choice: Logistic Regression preferred for simplicity

## 5. Optimisation

- Method: GridSearchCV
- Cross-validation: 5 folds
- Scoring: Accuracy
- Best parameters: {'classifier__C': 10}
- Best cross-validation accuracy: 1.0000

## 6. Final Training

After hyperparameter selection, the selected pipeline
was refitted on the complete dataset containing
200 records.

Saved artifact: models/best_model.pkl

The saved artifact includes text preprocessing,
numeric feature scaling, and the trained classifier.

## 7. Known Limitations

1. The dataset contains synthetic, template-generated examples.
2. Repeated templates and category-specific vocabulary
   may make the classification task easier than real-world use.
3. The previously recorded accuracy may be optimistic.
4. Cross-validation on this dataset does not establish
   generalisation to independently collected questions.
5. The model has not been validated on live user queries.
6. The model is a prototype and is not yet approved
   for production deployment.

## 8. Intended Use

This model is a learning and evaluation artifact for
the AI Capstone: Local AI Assistant with Ollama.

It should be tested on independently written questions
before being considered for real-world deployment.
