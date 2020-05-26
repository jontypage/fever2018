#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from tqdm import tqdm_notebook

prefix = '../data/domlin_fever/'


# In[4]:


train_labels = pd.read_table(prefix + 'train-domlin.label', header=None)
train_hypothesis = pd.read_table(prefix + 'train-domlin.hypothesis', header=None)
train_premise = pd.read_table(prefix + 'train-domlin.premise', header=None)
dev_labels = pd.read_table(prefix + 'dev-domlin.label', header=None)
dev_hypothesis = pd.read_table(prefix + 'dev-domlin.hypothesis', header=None)
dev_premise = pd.read_table(prefix + 'dev-domlin.premise', header=None)
test_hypothesis = pd.read_table(prefix + 'test-domlin.hypothesis', header=None)
test_premise = pd.read_table(prefix + 'test-domlin.premise', header=None)


# In[18]:

text_a_train = train_hypothesis[0].str.replace(r'\n', ' ', regex=True)
text_a_train = text_a_train.str.replace('-LRB-', '(')
text_a_train = text_a_train.str.replace('-RRB-', ')')
text_a_train = text_a_train.str.replace('-COLON-', ':')
text_b_train = train_premise[0].str.replace(r'\n', ' ', regex=True)
text_b_train = text_b_train.str.replace('-LRB-', '(')
text_b_train = text_b_train.str.replace('-RRB-', ')')
text_b_train = text_b_train.str.replace('-COLON-', ':')

train_df = pd.DataFrame({
    'id':range(len(train_labels)),
    'label':train_labels[0],
    'alpha':['a']*train_labels.shape[0],
    'text_a': text_a_train,
    'text_b': text_b_train
})

train_df.head()

print(train_df.label.value_counts())
count_class_1, count_class_0, count_class_2 = train_df.label.value_counts()
df_class_0 = train_df[train_df['label'] == 0]
df_class_1 = train_df[train_df['label'] == 1]
df_class_2 = train_df[train_df['label'] == 2]

df_class_1_under = df_class_1.sample(count_class_2, replace=True)
df_class_0_under = df_class_0.sample(count_class_2, replace=True)
train_df_new = pd.concat([df_class_0_under, df_class_1_under, df_class_2], axis=0).sample(frac=1)

# In[19]:

text_a_dev = dev_hypothesis[0].str.replace(r'\n', ' ', regex=True)
text_a_dev = text_a_dev.str.replace('-LRB-', '(')
text_a_dev = text_a_dev.str.replace('-RRB-', ')')
text_a_dev = text_a_dev.str.replace('-COLON-', ':')
text_b_dev = dev_premise[0].str.replace(r'\n', ' ', regex=True)
text_b_dev = text_b_dev.str.replace('-LRB-', '(')
text_b_dev = text_b_dev.str.replace('-RRB-', ')')
text_b_dev = text_b_dev.str.replace('-COLON-', ':')

dev_df = pd.DataFrame({
    'id':range(len(dev_labels)),
    'label':dev_labels[0],
    'alpha':['a']*dev_labels.shape[0],
    'text_a': text_a_dev,
    'text_b': text_b_dev
})

dev_df.head()

# In[20]:


text_a_test = test_hypothesis[0].str.replace(r'\n', ' ', regex=True)
text_a_test = text_a_test.str.replace('-LRB-', '(')
text_a_test = text_a_test.str.replace('-RRB-', ')')
text_a_test = text_a_test.str.replace('-COLON-', ':')
text_b_test = test_premise[0].str.replace(r'\n', ' ', regex=True)
text_b_test = text_b_test.str.replace('-LRB-', '(')
text_b_test = text_b_test.str.replace('-RRB-', ')')
text_b_test = text_b_test.str.replace('-COLON-', ':')

test_df = pd.DataFrame({
    'id':range(len(text_a_test)),
    'label':[1]*test_hypothesis.shape[0],
    'alpha':['a']*test_hypothesis.shape[0],
    'text_a': text_a_test,
    'text_b': text_b_test
})

test_df.head()


# In[21]:


train_df.to_csv('data_domlin3/train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
dev_df.to_csv('data_domlin3/dev.tsv', sep='\t', index=False, header=False, columns=dev_df.columns)
test_df.to_csv('data_domlin3/test.tsv', sep='\t', index=False, header=False, columns=test_df.columns)
