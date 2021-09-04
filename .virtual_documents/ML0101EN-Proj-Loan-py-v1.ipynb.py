import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic("matplotlib", " inline")

import seaborn as sns


get_ipython().getoutput("wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv")


df = pd.read_csv('loan_train.csv')
df.head()


df.shape


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


df['loan_status'].value_counts()











df["dayofname"] = df['effective_date'].dt.day_name()


df['dayofweek'] = df['effective_date'].dt.dayofweek

df["dayofweek"].head()


df["dayofweek"].value_counts()


df.head()





df_dayofname = df[["dayofname","loan_status"]]
df_dayofname.head()


groupby_dayname = df_dayofname.groupby(["dayofname","loan_status"]).size()
groupby_dayname


sns.countplot(x = "dayofname", hue = "loan_status", data = df_dayofname)


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


df[['Principal','terms','age','Gender','education']].head()





Feature["Bechalor"].value_counts()








X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]






































from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss



get_ipython().getoutput("wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv")


test_df = pd.read_csv('loan_test.csv')
test_df.head()









