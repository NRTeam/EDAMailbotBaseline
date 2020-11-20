# Prediction with stacked classifier for Services, Impact, Urgency and incident_types

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib


#df = pd.read_pickle('Pickle/2-Preparation/020-incidents_lemmatized.pkl')
df = pd.read_pickle('Pickle/2-Preparation/030-incidents_sampled_by_service.pkl')

vectorizer = TfidfVectorizer(sublinear_tf=True)
features = vectorizer.fit_transform(df.concatenated_lemmas_string)
X = features
#X_all = df.body_lemma_string
y_services_all = df.ServiceProcessed
y_urgencies_all = df.Urgency
y_impacts_all = df.Impact
y_incidenttypes_all = df.IncidentType

Y = np.vstack((y_services_all, y_urgencies_all, y_impacts_all, y_incidenttypes_all)).T

#classifier = RandomForestClassifier()
#classifier = SVC(probability=True)
#classifier = Perceptron()
#classifier = MultinomialNB(),
classifier = LogisticRegression(random_state=0)

multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
model = multi_target_classifier.fit(X, Y)

#save model to disk
filename = "Models/model-DE.model"
joblib.dump([model, vectorizer],filename)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_pickle('Pickle/2-Preparation/030-incidents_sampled_by_service.pkl')

X_all = df.concatenated_lemmas_string
y_all = df.ServiceProcessed

# aufteilen der Daten in einen Trainings- und Testdatensatz (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
vectorizer = TfidfVectorizer(sublinear_tf=True)


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

all_services = y_all.unique()
print("Count of Services:", len(all_services))

#clf = xgb.XGBClassifier()
#clf = RandomForestClassifier()
#clf = SVC(probability=True)
#clf = Perceptron()
#clf = MultinomialNB(),
clf = LogisticRegression(random_state=0)

y = df.ServiceProcessed
x = vectorizer.fit_transform(df.concatenated_lemmas_string)

y_pred = cross_val_predict(clf, x, y, cv=20)
conf_mat = confusion_matrix(y, y_pred)

plt.figure(figsize = (25,25))
sns.set(font_scale=0.75)
sns.heatmap(conf_mat, square=True, annot=True, fmt="d", annot_kws={"size": 7}, cbar=False, cmap="YlGnBu", linewidths=1.0, xticklabels=all_services, yticklabels=all_services)
plt.xlabel('Predicted service', fontsize = 16)
plt.ylabel('True service', fontsize = 16)
plt.show()

# Benchmark multiple classifiers

models = [
    xgb.XGBClassifier(),
    RandomForestClassifier(),
    SVC(),
    Perceptron(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]

y = 'ServiceProcessed'

labels = df[y]
features = vectorizer.fit_transform(df.concatenated_lemmas_string)

CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  f1s_weighted = cross_val_score(model, features, labels, scoring='f1_weighted', cv=CV)
  for fold_idx, f1 in enumerate(f1s_weighted):
    entries.append((model_name, fold_idx, f1))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1_weighted'])

sns.boxplot(x='model_name', y='f1_weighted', data=cv_df).set_title(y)
chart = sns.stripplot(x='model_name', y='f1_weighted', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
chart.set_xticklabels(
                  chart.get_xticklabels(), 
                  rotation=45, 
                  horizontalalignment='right')

plt.show()