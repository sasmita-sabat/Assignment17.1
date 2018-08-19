
# coding: utf-8

# In[2]:


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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(url)
titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

train_p = titanic.drop(["Cabin","PassengerId","Ticket","Name","Embarked","Fare"],axis=1)

train_p.head()
train_p.info()


#Age
#Logically, when accidents occur, we normally give priority to children and old people. 
#Therefore, I divide age into 3 groups:(,16),(16,60),(60,) to represent child, adult and old
#since most passengers are adults, we fill nulls with adult
#0:child;1:adult,2:old
train_p['Age'].dropna().hist(bins=70)
plt.show()

train_p['Age_new'] = 1
train_p['Age_new'][train_p["Age"]<16] = 0
train_p['Age_new'][train_p["Age"]>60] = 2
train_p['Age_new']=train_p['Age_new'].astype(int)
train_p['Age_new'].hist()
plt.show()        
train_p.info()



train_p['Family'] =  train_p["Parch"] + train_p["SibSp"]
train_p['Family'].loc[train_p['Family'] > 0] = 1
train_p['Family'].loc[train_p['Family'] == 0] = 0
train_p = train_p.drop(['SibSp','Parch'], axis=1)
#gender
#male is 1
train_p['Gender'] = 0
train_p['Gender'][train_p['Sex']=="male"] = 1

#final data preparation
train_y = train_p['Survived']
train_x = train_p.drop(['Survived','Age',"Sex"],axis=1)


train_y.head()
train_x.head()
train_x.info()

#decision tree model
model_tree = tree.DecisionTreeClassifier()
model_tree = model_tree.fit(train_x, train_y)

print(model_tree.feature_importances_)
print(model_tree.score(train_x, train_y))


