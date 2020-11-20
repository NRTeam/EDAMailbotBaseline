# Current Model Training for EDA Mailbot
This repository contains a base line for the FDFA, IPT and eestec LC Zurich Pocathon 'Sk[AI] is the limit! Hack it happen'. 

Currently we are using this code to train our model, which predicts services, impact, urgency and incident type of german mails.
The data you will receive is equal to the export after cleaning.

## Cleaning
- Read data from csv
- Reduce to relevant columns
- Remove signatures from Mail-Body
- Remove control-characters (like New-Line,…) from Mail-Body
- Run langdetect and export only german mails

## Preparation
- Concatenate Subject & Mail-Body, tokenize them 
- Group by Service and balance groups (at the moment for each group 80-100 Mails are taken)
- Fill up Eda_others 
- Save incidents sampled/ balanced by service

## Training

- Vectorize with TF-IDF
- Stack services, urgencies, impacts & incident-types on one model
- Use MultiOutputClassifier (currently: LogisticRegression)
- Save Model & print some graphs 
  - Confusion matrix with the current Classifier (Cross-Validated)
  - Box-Plot with f1 weighted score of different classifiers (XG-Boost, RandomForest, SVC, Perceptron, MultinominalNB, LogisticRegression)
