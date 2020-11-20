import pandas as pd
import string
import nltk
from germalemma import GermaLemma
from HanTa import HanoverTagger as ht
import math

df = pd.read_pickle('Pickle/1-Cleaning/060-cleaned_incidents_de.pkl')

tagger = ht.HanoverTagger('morphmodel_ger.pgz')

# durchlaufe Preprocess-Pipeline und verwende nur Nomen.
def preprocess(text):
    nouns = []
    try:
        # tokenize in sentences
        sentences = nltk.sent_tokenize(text, language='german')
        sentences_tok = [nltk.word_tokenize(sent, language='german') for sent in sentences]

        for sent in sentences_tok:
            tags = tagger.tag_sent(sent)
            nouns_from_sent = [lemma for (word, lemma, pos) in tags if pos == "NN" or pos == "NE"]
            nouns.extend(nouns_from_sent)
    except TypeError:
        pass
    except KeyError:
        print("KeyError")
        pass
    return " ".join(nouns)
# Verwende alle Worttypen
def preprocessFull(text):
    words = []
    try:
        # tokenize in sentences
        sentences = nltk.sent_tokenize(text, language='german')
        sentences_tok = [nltk.word_tokenize(sent, language='german') for sent in sentences]

        for sent in sentences_tok:
            tags = tagger.tag_sent(sent)
            words_from_sent = [lemma for (word, lemma, pos) in tags]
            words.extend(words_from_sent)    
    except TypeError:
        pass
    except KeyError:
        print("KeyError")
        pass
    return " ".join(words)

df['subject_lemma_string'] = df['MailSubject'].apply(preprocessFull)
df['body_lemma_string'] = df['MailTextBody'].apply(preprocessFull)

#df['subject_lemma_string'] = df['MailSubject'].apply(preprocess)
#df['body_lemma_string'] = df['MailTextBody'].apply(preprocess)

df['concatenated_lemmas_string'] = df['subject_lemma_string'] + df['body_lemma_string']

df.to_pickle('Pickle/2-Preparation/020-incidents_lemmatized.pkl')

df = pd.read_pickle('Pickle/2-Preparation/020-incidents_lemmatized.pkl')

df['concatenated_lemmas_string'] = df['subject_lemma_string'] + df['body_lemma_string']

mailsProcessedDf = df.groupby('ServiceProcessed')
pd.set_option('display.max_rows', None)
mailsProcessed = mailsProcessedDf.size().sort_values(ascending=False)
print(mailsProcessed)

threshold_min_value = 80

thresholded_target_services = list(df.groupby('ServiceProcessed').filter(lambda x: len(x) >= threshold_min_value).groupby('ServiceProcessed').groups.keys())
thresholded_target_services

## sampling / balancieren
# use thresholded target_services instead of creating a list manually as above
target_services = thresholded_target_services
threshold_max_value = 100
# define services to use for eda other & how many incidents to take per service
eda_other_targets = ['EDA_S_BA_Datenablage', 'EDA_S_BA_Internetzugriff', 'EDA_S_BA_RemoteAccess', 'EDA_S_IT Sicherheit', 'EDA_S_Netzwerk Ausland', 'EDA_S_Raumbewirtschaftung']
threshold_eda_other_max_value = math.floor(threshold_max_value / len(eda_other_targets))

# create a large df_other with all other services, in order to subsample from this one later
df_other = df[~df.ServiceProcessed.isin(target_services)]
# create an empty dataframe (could be done easier..)
df_other_sampled = df_other.reset_index(drop=True)
df_other_sampled = df_other_sampled[0:0] 
for eda_other_target in eda_other_targets:
    totalForService = df_other[df_other.ServiceProcessed == eda_other_target]["MailTextBody"].size
    print('total incidents of: ' + eda_other_target + ': ' + str(totalForService))
    if(totalForService > threshold_eda_other_max_value):
        df_other_sampled = pd.concat([df_other_sampled, df_other[df_other.ServiceProcessed == eda_other_target].sample(n=threshold_eda_other_max_value)])
    else:
        df_other_sampled = pd.concat([df_other_sampled, df_other[df_other.ServiceProcessed == eda_other_target]])

other_count = df_other_sampled['MailTextBody'].size
print('Total incidents in df_other_sampled: ' + str(other_count))
#filling up 
if(other_count < threshold_max_value):
     df_other_sampled = pd.concat([df_other_sampled, df_other[df_other['ServiceProcessed'].isin(eda_other_targets)].sample(n=(threshold_max_value - other_count))])
other_count = df_other_sampled['MailTextBody'].size
print('Total incidents in df_other_sampled after filling up: ' + str(other_count))

other_count = df_other_sampled['MailTextBody'].size
print('Total incidents in df_other_sampled: ' + str(other_count))

print('distribution in eda_other')
print(df_other_sampled.groupby('ServiceProcessed').size()) 

df_other_sampled.loc[:,'ServiceProcessed'] = 'EDA_other'

# create an empty dataframe (could be done easier..)
df_sampled = df_other_sampled.groupby('ServiceProcessed').apply(pd.DataFrame.sample, n=threshold_max_value).reset_index(drop=True)
df_sampled = df_sampled[0:0] 

for target_service in target_services:
    if(df[df.ServiceProcessed == target_service]['MailTextBody'].size > threshold_max_value):
        df_sampled = pd.concat([df_sampled, df[df.ServiceProcessed == target_service].sample(n=threshold_max_value)])
    else:
        df_sampled = pd.concat([df_sampled, df[df.ServiceProcessed == target_service]])

# add a subsampling from df_other for the EDA_other service
df_sampled = pd.concat([df_sampled, df_other_sampled])

# reset technical dataframe indexes newly
df_sampled = df_sampled.reset_index(drop=True)

# print sizes
print(df_sampled.groupby('ServiceProcessed').size()) 

df = df_sampled.copy()

df.to_pickle('Pickle/2-Preparation/030-incidents_sampled_by_service.pkl')