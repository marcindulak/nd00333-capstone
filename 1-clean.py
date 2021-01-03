#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
  from google.colab import drive
  drive.mount('/content/drive')
  drive_path = '/content/drive/My\ Drive/'
except ImportError:
  drive_path = '.'


# In[2]:


from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns


# In[3]:


dataset_name = 'cse-cic-ids2018'
dataset_file = f'{dataset_name}.zip'
dataset_path = Path(drive_path, f'datasets/registry.opendata.aws/{dataset_name}')
get_ipython().system('pwd')
print(dataset_path)


# Download the https://www.unb.ca/cic/datasets/ids-2018.html dataset from s3 https://registry.opendata.aws/cse-cic-ids2018/

# In[4]:


get_ipython().system('aws s3 sync --no-sign-request --region eu-central-1      "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"      $dataset_path')


# In[5]:


get_ipython().system('cd $dataset_path && zip $dataset_file *.csv')


# Create local directory to store the dataset files.

# In[6]:


get_ipython().system(' if ! test -r $dataset_name; then mkdir $dataset_name && cp $dataset_path/$dataset_file $dataset_name; fi')


# In[7]:


get_ipython().system('ls -al $dataset_name')


# In[8]:


get_ipython().system(' if test -r $dataset_name/$dataset_file; then cd $dataset_name && unzip $dataset_file && rm -f $dataset_file; fi')


# In[9]:


get_ipython().system('ls -al $dataset_name')


# In[10]:


get_ipython().system('ls -alh $dataset_name')


# In[11]:


get_ipython().system('df -h')


# Perform cleaning and feature selection separately for every data file

# In[12]:


from nd00333.dataset.clean import clean


# In[13]:


import importlib


# In[14]:


importlib.reload(clean)


# Summarize one of the smaller data sets.
# 
# The following observations can be made:
# 
# 1. 'Flow Byts/s' and 'Flow Pkts/s' columns contain non-numeric values
# 2. 'Init Fwd Win Byts' and 'Init Bwd Win Byts' contain a negative number '-1'
# 3. 'Flow IAT Min' amd 'Fwd IAT Min' contain large absolute negative values
# 
# The rows with those values in the respective columns will be removed (1. and 2., note that 2. results in a significant decrease in the number of non-Benign flows for a couple of data sets, e.g. for 'DoS attacks-Hulk', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP') or replaced (3.) in the the `get_clean_df` function.

# In[15]:


df = pd.read_csv(f'{dataset_name}/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
df = clean.get_clean_df(df, verbose=2)
df['target'] = df.pop('Label')
feature_list = clean.get_feature_list(df, tolerance=0.0001, sample_fraction=0.5)
print(feature_list)
del df


# A 16 GB machine is unable to keep copies of the largest dataset `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` in memory. Therefore some of the low variance and duplicate features found in smaller datasets are removed upfront from the lagest dataset to reduce its size. Moreover the lagest data file contains `extra_features` not present in other data files, and they are therefore removed. Additionally, due to a large number (almost 8 millions) samples if the largest data set a sample of 5% (instead of 50% as in all other data files) is used in the process of feature selection.

# Many people (e.g. Frank Harrell https://twitter.com/f2harrell/status/1137012097391312897?lang=en `Feature selection doesn't work in general because it can't find the right variables and distorts statistical properties.  One summary of the evils of stepwise`) claim that no feature selection should be performed. In this case reducing the number of features is necessary due to limited computing resources.
# 
# In principle a feature selection should happen on an isolated subset of the data, in order to not involve the test data in any model choices. This approach is not followed strictly here, as the feature selection is performed based on the full dataset, but this is acceptable, since another separate test set https://www.unb.ca/cic/datasets/ids-2017.html is used for the final estimation of the model performance.
# 
# The features are selected in `get_feature_list` using an addition process, where features are added on-by-one in the order of importance, only if by adding a feature the performance metrics (the macro average of recall across all target classes) increases by a threshold.

# In[16]:


columns = []
for dataset_file in sorted(glob(f'{dataset_name}/*.csv')):
  columns_dataset_file = pd.read_csv(f'{dataset_file}', index_col=0, nrows=0).columns.tolist()
  columns_new = set(columns_dataset_file) - set(columns)
  if len(columns_new):
    print(f'New columns in {dataset_file}', columns_new)
    columns.extend(columns_new)


# In[17]:


quasi_constant_features = ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg',
                           'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
duplicated_features = ['Subflow Fwd Pkts', 'Subflow Bwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Byts', 'Fwd Seg Size Avg',
                       'Bwd Seg Size Avg', 'SYN Flag Cnt', 'ECE Flag Cnt']
extra_features = ['Src IP', 'Src Port', 'Dst Port', 'Dst IP']


# In[18]:


selected_features = {}
for dataset_file in sorted(glob(f'{dataset_name}/*.csv')):
  print('#' * 80)
  print('New datafile:', dataset_file)
  print('#' * 80)
  if dataset_file == f'{dataset_name}/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv':
    columns_to_remove = quasi_constant_features + duplicated_features + extra_features
    sample_fraction = 0.05
  else:
    columns_to_remove = extra_features
    sample_fraction = 0.5
  columns = pd.read_csv(dataset_file, index_col=0, nrows=0).columns.tolist()
  usecols = []
  for column in columns:
    if column not in columns_to_remove:
      usecols.append(column)    
  df = pd.read_csv(dataset_file, usecols=usecols)
  df = clean.get_clean_df(df, verbose=1)
  df['target'] = df.pop('Label')
  feature_list = clean.get_feature_list(df, tolerance=0.001, sample_fraction=sample_fraction)
  del df
  selected_features[dataset_file] = feature_list


# Find the union set of selected features across all data files

# In[19]:


selected_features_common = []
for dataset_file, features_list in sorted(selected_features.items()):
  print(f'Merging features for {dataset_file}', features_list)
  for feature in features_list:
    if feature not in selected_features_common:
      selected_features_common.append(feature)


# In[20]:


print(f'Number of selected features {len(selected_features_common)}')
for feature in selected_features_common:
  print(feature)


# Save selected features data into new csv files

# In[21]:


dataset_name_clean = dataset_name + '-clean'


# In[22]:


get_ipython().system('mkdir -p $dataset_name_clean')


# In[24]:


for dataset_file in sorted(glob(f'{dataset_name}/*.csv')):
  file_name = dataset_file.split('/')[-1]
  print('#' * 80)
  print('New datafile:', dataset_file)
  print('#' * 80)
  df = pd.read_csv(dataset_file, usecols=selected_features_common + ['Label'])
  df = clean.get_clean_df(df, verbose=1)  
  df.to_csv(f'{dataset_name_clean}/{file_name}', index=False)


# In[25]:


get_ipython().system('ls -al $dataset_name_clean')


# In[26]:


get_ipython().system('ls -alh $dataset_name_clean')


# In[27]:


get_ipython().system('head -2 $dataset_name_clean/*.csv')


# Load all data files into a common dataframe

# In[28]:


df = pd.concat(map(pd.read_csv, glob(f'{dataset_name_clean}/*.csv')))


# In[29]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df.head().transpose())
    display(df.describe().transpose())


# In[30]:


size = df.groupby(['Label']).size().reset_index(name='count')
display(size)


# In[31]:


size['fraction'] = (df.groupby(['Label']).size()
                    .reset_index(name='count').apply(lambda x: x['count'] / df.shape[0], axis=1))
display(size)


# In[32]:


size.plot.bar(x='Label', y='fraction')


# Explore correlations between features. In can be noted that there are several groups of highly correlated features (abs(corr)>=0.9), for example:
# 
# 1. 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Bwd Pkt Len Std', 'Bwd Header Len', and 'Pkt Len Max'
# 2. 'Pkt Len Std', and 'Pkt Len Max'
# 3. 'Flow Pkts/s', 'Flow Duration', and 'Flow IAT Max'
# 4. 'RST Flag Cnt' and 'ECE Flag Cnt'
# 
# These correlations, for the non-Benign labels are explored in more details further below, and since the plots show that the correlation coefficient does not represent a linear relationship, all the above features are kept.

# In[33]:


df_corr = df.corr(method='spearman')
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_corr, xticklabels=df_corr.columns, yticklabels=df_corr.columns, annot=True, fmt='.1f', ax=ax)


# In[34]:


def plot_corr(data, x, y, xlim, ylim):
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(15, 15))
  data.plot.scatter(x=x, y=y, ax=ax)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)


# In[35]:


plot_corr(df[df['Label'] != 'Benign'], 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', xlim=(-1, 1e5), ylim=(-1, 3e3))


# In[36]:


plot_corr(df[df['Label'] != 'Benign'], 'Pkt Len Std', 'Pkt Len Max', xlim=(-1, 1e3), ylim=(-1, 3e3))


# In[37]:


plot_corr(df[df['Label'] != 'Benign'], 'Flow Pkts/s', 'Flow Duration', xlim=(-1, 1e4), ylim=(-1, 1e4))


# In[38]:


plot_corr(df[df['Label'] != 'Benign'], 'RST Flag Cnt', 'ECE Flag Cnt', xlim=(-1, 2), ylim=(-1, 2))


# Save the clean dataset archive

# In[39]:


dataset_clean_file = f'{dataset_name_clean}.zip'


# In[40]:


get_ipython().system('rm -f $dataset_clean_file')
get_ipython().system('zip -r $dataset_clean_file $dataset_name_clean')


# In[41]:


get_ipython().system(' /bin/cp -f $dataset_clean_file $dataset_path')


# In[43]:


get_ipython().system("jupyter nbconvert --to html '1-clean.ipynb'")

