# Current Model Training for EDA Mailbot
This repository contains a base line for the FDFA & IPT Pocathon with the ETH. 

This training consists of following 3 Parts:

## Cleaning
- Read data from csv
- Reduce to relevant columns
- Remove signatures from Mail-Body
- Remove control-characters (like New-Line,…) from Mail-Body
- Run langdetect and export only german mails

## Preparation
- Read data from csv
- Reduce to relevant columns
- Remove signatures from Mail-Body
- Remove control-characters (like New-Line,…) from Mail-Body
- Run langdetect and export only german mails

## Training

- Vectorize with TF-IDF
- Stack services, urgencies, impacts & incident-types on one model
- Use MultiOutputClassifier (currently: LogisticRegression)
- Save Model & print some graphs 
  - Confusion matrix with the current Classifier (Cross-Validated)
  - Box-Plot with f1 weighted score of different classifiers (XG-Boost, RandomForest, SVC, Perceptron, MultinominalNB, LogisticRegression)
