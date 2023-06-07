#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

import numpy as np
#import os,sys
from scipy import stats

# calculate accuracy measures and confusion matrix
from sklearn import metrics


# # Importing Dataset

# In[2]:


dataset=pd.read_csv("bank-full.csv")


# # EDA

# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


dataset.shape


# In[6]:


#checking the datat type
dataset.dtypes


# In[7]:


# Checking for null values
val=dataset.isnull().values.any()

if val==True:
    print("Missing values present : ", dataset.isnull().values.sum())
    dataset=dataset.dropna()
else:
    print("No missing values present")


# In[8]:


# Finding the summary of the data
dataset.describe()


# In[9]:


dataset.skew()


# In[10]:


dataset.median()

dataset.std()


# # Checking the presence of outliers

# In[11]:


# AGE

print('Max age: ', dataset['age'].max())
print('Min age: ', dataset['age'].min())


# In[12]:


plt.figure(figsize = (30,12))
sns.countplot(x = 'age',  palette="rocket", data = dataset)
plt.xlabel("Age", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Age Distribution', fontsize=15)


# In[13]:


sns.boxplot(x = 'age', data = dataset, orient = 'v')
plt.ylabel("Age", fontsize=15)
plt.title('Age Distribution', fontsize=15)


# In[14]:


#Quartiles
print('1º Quartile: ', dataset['age'].quantile(q = 0.25))
print('2º Quartile: ', dataset['age'].quantile(q = 0.50))
print('3º Quartile: ', dataset['age'].quantile(q = 0.75))


# In[15]:


# Interquartile range, IQR = Q3 - Q1
 # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
 # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
   
print('Ages above: ', dataset['age'].quantile(q = 0.75) + 
                     1.5*(dataset['age'].quantile(q = 0.75) - dataset['age'].quantile(q = 0.25)), 'are outliers')


# In[16]:


print('Numerber of outliers: ', dataset[dataset['age'] > 70.5]['age'].count())
print('Number of clients: ', len(dataset))
#Outliers in %
print('Outliers are:', round(dataset[dataset['age'] > 70.5]['age'].count()*100/len(dataset),2), '%')


# In[17]:


#Job
plt.figure(figsize = (30,12))
sns.countplot(x = 'job',data = dataset)
plt.xlabel("job", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Job Distribution', fontsize=20)


# In[18]:


#Marital

#plt.figure(figsize = (30,12))
sns.countplot(x = 'marital',data = dataset)
plt.xlabel("Marital", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Marital Distribution', fontsize=15)


# In[19]:


sns.boxplot(x='marital',y='age',hue='Target',data=dataset)


# In[20]:


#Education

#plt.figure(figsize = (30,12))
sns.countplot(x = 'education',data = dataset)
plt.xlabel("Education", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Education Distribution', fontsize=15)


# In[21]:


sns.boxplot(x='education',y='age',hue='Target',data=dataset)


# In[22]:


#plt.figure(figsize = (30,12))
sns.countplot(x = 'default',data = dataset)
plt.xlabel("Default", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Default Distribution', fontsize=15)


# In[23]:


sns.boxplot(x='default',y='age',hue='Target',data=dataset)


# In[24]:


print('Default:\n No credit in default:'     , dataset[dataset['default'] == 'no']     ['age'].count(),
              '\n Yes to credit in default:' , dataset[dataset['default'] == 'yes']    ['age'].count())


# In[25]:


# Housing loan

#plt.figure(figsize = (30,12))
sns.countplot(x = 'housing',data = dataset)
plt.xlabel("Housing", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Housing Distribution', fontsize=15)


# In[26]:


print('Housing:\n No Housing:'     , dataset[dataset['housing'] == 'no']     ['age'].count(),
              '\n Yes Housing:' , dataset[dataset['housing'] == 'yes']    ['age'].count())


# In[27]:


sns.boxplot(x='housing',y='age',hue='Target',data=dataset)


# In[28]:


# Loan

#plt.figure(figsize = (30,12))
sns.countplot(x = 'loan',data = dataset)
plt.xlabel("Loan", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Loan Distribution', fontsize=15)


# In[29]:


print('Loan:\n No Personal loan:'     , dataset[dataset['loan'] == 'no']     ['age'].count(),
              '\n Yes Personal Loan:' , dataset[dataset['loan'] == 'yes']    ['age'].count())


# In[30]:


sns.boxplot(x='loan',y='age',hue='Target',data=dataset)


# In[31]:


#plt.figure(figsize = (30,12))
sns.countplot(x = 'contact',data = dataset)
plt.xlabel("Contact", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Contact Distribution', fontsize=15)


# In[32]:


print('Contact:\n Unknown Contact:'     , dataset[dataset['contact'] == 'unknown']     ['age'].count(),
              '\n Cellular Contact:'   , dataset[dataset['contact'] == 'cellular']    ['age'].count(),
              '\n Telephone Contact:'  , dataset[dataset['contact'] == 'telephone']   ['age'].count())


# In[33]:


#Month

#plt.figure(figsize = (30,12))
sns.countplot(x = 'month',data = dataset)
plt.xlabel("In which Month was a person contacted", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Monthly Distribution', fontsize=15)


# In[34]:


#Day

sns.boxplot(x=dataset["day"])


# In[35]:


#Duration


# In[36]:


sns.boxplot(x=dataset["duration"])


# In[37]:


# Quartiles
print('1º Quartile: ', dataset['duration'].quantile(q = 0.25))
print('2º Quartile: ', dataset['duration'].quantile(q = 0.50))
print('3º Quartile: ', dataset['duration'].quantile(q = 0.75))
print('4º Quartile: ', dataset['duration'].quantile(q = 1.00))


# In[38]:


# Interquartile range, IQR = Q3 - Q1
# lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
# Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
  
print('Duration above: ', dataset['duration'].quantile(q = 0.75) + 
                    1.5*(dataset['duration'].quantile(q = 0.75) - dataset['duration'].quantile(q = 0.25)), 'are outliers')


# In[39]:


print('Numerber of outliers: ', dataset[dataset['duration'] > 643.0]['duration'].count())
print('Number of clients: ', len(dataset))
#Outliers in %
print('Outliers are:', round(dataset[dataset['duration'] > 643.0]['duration'].count()*100/len(dataset),2), '%')


# In[40]:


#Campaign


# In[41]:


plt.figure(figsize = (30,12))
sns.countplot(x = 'campaign', data = dataset)
plt.xlabel("Campaign", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Campaign Distribution', fontsize=15)


# In[42]:


sns.boxplot(x = 'campaign', data = dataset, orient = 'v')
plt.ylabel("Campaign", fontsize=15)
plt.title('Campaign Distribution', fontsize=15)


# In[43]:


# Quartiles
print('1º Quartile: ', dataset['campaign'].quantile(q = 0.25))
print('2º Quartile: ', dataset['campaign'].quantile(q = 0.50))
print('3º Quartile: ', dataset['campaign'].quantile(q = 0.75))
print('4º Quartile: ', dataset['campaign'].quantile(q = 1.00))


# In[44]:


# Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Campaign above: ', dataset['campaign'].quantile(q = 0.75) + 
                      1.5*(dataset['campaign'].quantile(q = 0.75) - dataset['campaign'].quantile(q = 0.25)), 'are outliers')


# In[45]:


print('Numerber of outliers: ', dataset[dataset['campaign'] > 6.0]['campaign'].count())
print('Number of clients: ', len(dataset))
#Outliers in %
print('Outliers are:', round(dataset[dataset['campaign'] > 6.0]['campaign'].count()*100/len(dataset),2), '%')


# In[46]:


#pdays

sns.boxplot(x = 'pdays', data = dataset, orient = 'v')
plt.ylabel("pdays", fontsize=15)
plt.title('pdays Distribution', fontsize=15)


# In[47]:


#previous outcome

sns.countplot(x = 'poutcome', data = dataset, orient = 'v')
plt.ylabel("Poutcome", fontsize=15)
plt.title('Poutcome distribution', fontsize=15)


# In[48]:


print('poutcome:\n Unknown poutcome:'     , dataset[dataset['poutcome'] == 'unknown']   ['age'].count(),
              '\n Failure in  poutcome:'  , dataset[dataset['poutcome'] == 'failure']   ['age'].count(),
              '\n Other poutcome:'        , dataset[dataset['poutcome'] == 'other']     ['age'].count(),
              '\n Success in poutcome:'   , dataset[dataset['poutcome'] == 'success']   ['age'].count())


# In[49]:


sns.boxplot(x='poutcome',y='age',hue='Target',data=dataset)


# In[50]:


sns.countplot(x = 'Target', data = dataset, orient = 'v')
plt.ylabel("Target", fontsize=15)
plt.title('Target distribution', fontsize=15)


# In[51]:


#Calculate correlation matrix


# In[52]:


cor=dataset.corr()
cor


# In[53]:


plt.subplots(figsize=(10,8))
sns.heatmap(cor,annot=True)


# In[54]:


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[55]:


dataset=pd.get_dummies(dataset, columns=['job','marital','education','default','housing','loan','contact','day','month','poutcome'])


# In[56]:


dataset.head()


# In[57]:


x=dataset.drop('Target', axis=1)
y=dataset['Target']


# In[58]:


dataset.info()


# In[59]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[60]:


print(y)


# In[61]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


# In[62]:


print(x_train)


# In[63]:


print(x_test)


# In[64]:


print(y_train)


# In[65]:


print(y_test)


# In[66]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[67]:


print(x_train)
print(x_test)
print(y_train)
print(y_test)


# In[68]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[69]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[70]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

