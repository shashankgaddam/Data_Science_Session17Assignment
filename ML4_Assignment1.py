
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report
Url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(Url)
titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic.head()


# In[32]:


#drop passenger ID, Name, Ticket, Cabin#drop pas 
titanic.set_index('PassengerId', drop=True, inplace=True)
titanic.head()


# In[33]:


del titanic['Name']
del titanic['Cabin']
titanic['Embarked'].dropna(inplace=True)
titanic_df = titanic.copy(deep=True)
titanic.head()


# In[34]:


def gender(st):
    if st == 'male':
        return 1
    else:
        return 2
titanic['gender'] = titanic.Sex.apply(gender)
titanic.set_index('Sex', drop=True, inplace=True)
titanic.head()


# In[35]:


survived = titanic[titanic['Survived']==1]
surv_avg = survived.mean()['Age']

not_survived = titanic[titanic['Survived']==0]
nsurv_avg = not_survived.mean()['Age']

def fillavg(survv):
    if survv == 1:
        return surv_avg
    else:
        return nsurv_avg

titanic['avg'] = titanic.Survived.apply(fillavg)
titanic.head()


# In[36]:


titanic.Age.fillna(titanic['avg'], inplace = True)
titanic.describe()


# In[38]:


del titanic['avg']
titanic.head()


# In[40]:


from sklearn import datasets
from sklearn import tree
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
survived = titanic['Survived']
del titanic['Survived']
del titanic['Embarked']


# In[41]:


del titanic['Ticket']
from sklearn import cross_validation
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(titanic, survived, test_size = 0.2)


# In[42]:


from sklearn import tree
#Decision Tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_Train, Y_Train)
clf1.score(X_Test, Y_Test)


# In[43]:


predictions = clf1.predict(X_Test)
import sklearn.metrics
print(sklearn.metrics.confusion_matrix(Y_Test,predictions))
print(sklearn.metrics.accuracy_score(Y_Test, predictions))


# In[44]:


def map_data(df):
    # survived map
    survived_map = {0: False, 1: True}
    df['Survived'] = df['Survived'].map(survived_map)

    # PClass map
    pclass_map = {1: 'Upper Class', 2: 'Middle Class', 3: 'Lower Class'}
    df['Pclass'] = df['Pclass'].map(pclass_map)

    # Embarkation port map
    port_map = {'S': 'Southampton', 'C': 'Cherbourg','Q':'Queenstown'}
    df['Embarked'] = df['Embarked'].map(port_map)
    
    # add new column (FamilySize) to dataframe - sum of SibSp and Parch
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df
titanic_df = map_data(titanic_df)
titanic_df.head(3)


# In[45]:


#survival by Sex
#H0 = Gender has no impact on survivability
#HA = Gender does impact the chances of survivabily
table = pd.crosstab(titanic_df['Survived'],titanic_df['Sex'])
print(table)


# In[46]:


print(titanic_df.groupby('Sex').Survived.mean())


# In[47]:


from scipy import stats

chi2, p, dof, expected = stats.chi2_contingency(table.values)
results = [
    ['Item','Value'],
    ['Chi-Square Test',chi2],
    ['P-Value', p]
]
print(chi2, p)
#As the P-Value is less than 0.05 the probability of that the age group will impact the chances of survival is high. 
#Therefore we can reject the null hypothesis


# In[48]:


#survior by class
#H0 = Social Class has no impact on survivability
#HA = Social Class does impact the chances of survivabily
table = pd.crosstab(titanic_df['Survived'],titanic_df['Pclass'])
print(table)


# In[49]:


print(titanic_df.groupby('Pclass').Survived.mean())


# In[50]:


table = pd.crosstab([titanic_df['Survived']], titanic_df['Pclass'])
chi2, p, dof, expected = stats.chi2_contingency(table.values)
results = [
    ['Item','Value'],
    ['Chi-Square Test',chi2],
    ['P-Value', p]
]
print(chi2, p)
#As the P-Value is less than 0.05 the probability of that the age group will impact the chances of survival is high. 
#Therefore we can reject the null hypothesis


# In[51]:


# by age group
#H0 = Age Group has no impact on survivability
#HA = Age Group does impact the chances of survivabily
age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
titanic_df['age_group'] = pd.cut(titanic_df.Age, range(0, 81, 10), right=False, labels=age_labels)
print(titanic_df.groupby(['age_group']).Survived.mean())


# In[52]:


print(titanic_df.groupby(['Sex','age_group']).Survived.mean())


# In[53]:


table = pd.crosstab([titanic_df['Survived']], titanic_df['age_group'])
chi2, p, dof, expected = stats.chi2_contingency(table.values)
results = [
    ['Item','Value'],
    ['Chi-Square Test',chi2],
    ['P-Value', p]
]

print(chi2, p)
#As the P-Value is less than 0.05 the probability of that the age group will impact the chances of survival is high. 
#Therefore we can reject the null hypothesis

