
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm


# In[ ]:


# os.chdir('Desktop/Project_1')


# In[ ]:


train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')


# In[ ]:


train['area code'] = train['area code'].astype(object)


# In[ ]:


for i in range(train.shape[0]) :
    temp = train['phone number'][i].split('-')
    train['phone number'][i] = temp[0]


# In[ ]:


for i in train.columns :
    if (train[i].dtypes == 'object')  :
        print(i)
        train[i] = pd.Categorical(train[i])
        train[i] = train[i].cat.codes


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train['total day calls'])


# In[ ]:


cnames = ['account length','number vmail messages','total day minutes','total day calls','total day charge',
          'total eve minutes','total eve calls','total eve charge','total night minutes','total night calls',
          'total night charge','total intl minutes','total intl calls','total intl charge',
          'number customer service calls']


# In[ ]:


for i in cnames :
    print(i)
    q75, q25 = np.percentile(train.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    train = train.drop(train[train.loc[:,i] < min].index)
    train = train.drop(train[train.loc[:,i] > max].index)


# In[ ]:


df_corr = train.loc[:,cnames]
f, ax = plt.subplots(figsize = (7,5))
corr = df_corr.corr()

sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220,10,as_cmap = True),
           square = True, ax = ax)


# In[ ]:


cat_names = ['state','area code','phone number','international plan','voice mail plan']

for i in cat_names :
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train['Churn'],train[i]))
    print(p)


# In[ ]:


train = train.drop(['area code', 'phone number','total day minutes','total eve minutes',
                   'total night minutes', 'total intl minutes'], axis = 1)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(train['account length'], bins = 'auto')


# In[ ]:


cnames = ['account length', 'number vmail messages', 'total day calls', 'total day charge',
       'total eve calls', 'total eve charge', 'total night calls',
       'total night charge', 'total intl calls', 'total intl charge',
       'number customer service calls']


# In[ ]:


for i in cnames :
    train[i] = (train[i] - train[i].min())/(train[i].max() - train[i].min())


# In[ ]:


test['area code'] = test['area code'].astype(object)
for i in range(test.shape[0]) :
    temp = test['phone number'][i].split('-')
    test['phone number'][i] = temp[0]

for i in test.columns :
    if (test[i].dtypes == 'object')  :
        print(i)
        test[i] = pd.Categorical(test[i])
        test[i] = test[i].cat.codes
test = test.drop(['area code', 'phone number','total day minutes','total eve minutes',
                   'total night minutes', 'total intl minutes'], axis = 1)
cnames = ['account length', 'number vmail messages', 'total day calls', 'total day charge',
       'total eve calls', 'total eve charge', 'total night calls',
       'total night charge', 'total intl calls', 'total intl charge',
       'number customer service calls']
for i in cnames :
    test[i] = (test[i] - test[i].min())/(test[i].max() - test[i].min())


# In[ ]:


train['Churn'] = train['Churn'].replace(1,'yes')
train['Churn'] = train['Churn'].replace(0,'no')

x_train = train.values[:,0:14]
y_train = train.values[:,14]


# In[ ]:


test['Churn'] = test['Churn'].replace(1,'yes')
test['Churn'] = test['Churn'].replace(0,'no')

x_test = test.values[:,0:14]
y_test = test.values[:,14]


# In[ ]:


clf = tree.DecisionTreeClassifier(criterion = 'entropy').fit(x_train,y_train)
# clf = tree.DecisionTreeClassifier(criterion = 'gini').fit(x_train,y_train)
dt_pred = clf.predict(x_test)

CM = pd.crosstab(y_test,dt_pred)
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

CM
(FN*100)/(FN+TP)
accuracy_score(y_test,dt_pred)*100


# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 1).fit(x_train,y_train)
knn_pred = knn_model.predict(x_test)

CM = pd.crosstab(y_test,knn_pred)
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

CM
(FN*100)/(FN+TP)
#accuracy_score(y_test,knn_pred)*100


# In[ ]:


nb_model = GaussianNB().fit(x_train,y_train)
nb_pred = nb_model.predict(x_test)


CM = pd.crosstab(y_test,nb_pred)
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

(FN*100)/(FN+TP)

#accuracy_score(y_test,nb_pred)*100


# In[ ]:


rf_model = RandomForestClassifier(n_estimators = 100).fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

CM = pd.crosstab(y_test,nb_pred)
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

(FN*100)/(FN+TP)
accuracy_score(y_test,nb_pred)*100


# In[ ]:


train['Churn'] = train['Churn'].replace('no',0)
train['Churn'] = train['Churn'].replace('yes',1)

train_logit = pd.DataFrame(train['Churn'])
test_logit = pd.DataFrame(test['Churn'])

cnames = ['account length','number vmail messages','total day calls','total day charge',
          'total eve calls','total eve charge','total night calls',
          'total night charge','total intl calls','total intl charge',
          'number customer service calls']
train_logit = train_logit.join(train[cnames])
test_logit = test_logit.join(test[cnames])

cat_names = ['state','international plan', 'voice mail plan']
for i in cat_names :
    temp = pd.get_dummies(train[i], prefix = i)
    train_logit = train_logit.join(temp)
    temp = pd.get_dummies(test[i], prefix = i)
    test_logit = test_logit.join(temp)

train_cols = train.columns[0:14]
test_cols = test.columns[0:14]

logit = sm.Logit(train['Churn'],train[train_cols]).fit()

test['predict_prob'] = logit.predict(test[test_cols])
test['predict_val'] = 1
test.loc[test.predict_prob < 0.5, 'predict_val'] = 0

CM = pd.crosstab(test['Churn'],test['predict_val'])
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

(FN*100)/(FN+TP)
#accuracy_score(y_test,nb_pred)*100

