import pandas as pd
import regex
import re

cleanedDataPath = "Files/incidents_2020-11-05T071711Z.csv"

df = pd.read_csv(cleanedDataPath, sep=";", encoding="utf8")
df = df.dropna(subset=['MailTextBody'])
df = df.reset_index(drop=True)

df = df.sort_values(by='Id', ascending=False)

df.to_pickle('Pickle/1-Cleaning/010-loaded_incidents.pkl')
print("Number of Incidents loaded: " + str(df.Id.size))
df.head()



df = pd.read_pickle('Pickle/1-Cleaning/010-loaded_incidents.pkl')
dfreduced = pd.DataFrame(df, columns=['Id', 'Priority', 'Impact', 'Urgency', 'Service', 'ServiceProcessed', 'SupportGroup', 'IncidentType', 'MailSubject', 'MailAdresseFrom', 'MailDisplayNameFrom', 'MailTextBody', 'ConfirmStatus'])

dfreduced.to_pickle('Pickle/1-Cleaning/020-loaded_incidents_reducedColumns.pkl')


df = pd.read_pickle('Pickle/1-Cleaning/020-loaded_incidents_reducedColumns.pkl')

def getVornameFromMail(adresseFrom):
    try:
        return re.search('(\w*)-?\w*?\.(.*)@.*', adresseFrom.lower()).group(1)
    except AttributeError:
        return ""

def getNachnameFromTo(nameFrom):
    try:
        return re.search('(\w*)[-\w*]* \w*.* eda .*', nameFrom.lower()).group(1)
    except AttributeError:
        return ""

df['Vorname'] = df['MailAdresseFrom'].apply(getVornameFromMail)
df['Nachname'] = df['MailDisplayNameFrom'].apply(getNachnameFromTo)

maxsize = 10000000
for i, f in df.iterrows():

    pos = maxsize
    for p in [f.MailTextBody.lower().find(str(f.Vorname)), f.MailTextBody.lower().find(str(f.Nachname))]:
        if p > 0 and p < pos:
            pos = p
    if pos < maxsize and pos > 0:
        df.at[i, 'MailTextBody'] = f.MailTextBody[:pos]
    if pos == 0:
        df.at[i, 'MailTextBody'] = f.MailTextBody[:400]

df.to_pickle('Pickle/1-Cleaning/030-incidents_truncatedMailtext.pkl')
pd.set_option('display.max_colwidth',200)
pd.set_option('display.max_columns', None)
df.head()


# Clean text

df = pd.read_pickle('Pickle/1-Cleaning/030-incidents_truncatedMailtext.pkl')

def beautify_text(text):
    text = text.replace('\\r',' ')
    text = text.replace('\\n','')
    text = text.replace('\r',' ')
    text = text.replace('\n','')
    text = text.replace('  ',' ')
    text = text.lower()
    text = text.replace('&nbsp;','')
    return text.strip()

df['MailTextBody'] = df['MailTextBody'].apply(beautify_text)
df.to_pickle('Pickle/1-Cleaning/040-incidents-beautifiedMailtext.pkl')
df.head()

df = pd.read_pickle('Pickle/1-Cleaning/040-incidents-beautifiedMailtext.pkl')

#Language detection

from langdetect import detect

def detectLanguage(x):
    try:
        return detect(x)
    except:
        pass

df['MailLanguage'] = df['MailTextBody'].apply(detectLanguage)
df.to_pickle('Pickle/1-Cleaning/050-incidents_language.pkl')

df = pd.read_pickle('Pickle/1-Cleaning/050-incidents_language.pkl')
languageGroupedDf = df.groupby('MailLanguage')
pd.set_option('display.max_rows', None)
mailsByLanguage = languageGroupedDf.size().sort_values(ascending=False)
print(mailsByLanguage.nlargest(n=5))

df.head()

#Export German Mails
df = pd.read_pickle('Pickle/1-Cleaning/050-incidents_language.pkl')

#Reduce to german
df_de = df[(df['MailLanguage'] == 'de')]
df_de.to_pickle('Pickle/1-Cleaning/060-cleaned_incidents_de.pkl')