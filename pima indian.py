
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[22]:


colnames = ['preg','plas','pres','skin','test','mass','pedi','age','class']


# In[3]:


import pandas as pd
from sklearn.model_selection import cross_val_score


# In[15]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"


# In[23]:


df=pd.read_csv("C:\\Users\\Vaibhav\\Desktop\\pima.csv",names=colnames,header=None)


# In[19]:


#df=pd.read_csv("C:\\Users\\Vaibhav\\Downloads\\pima.csv")


# In[24]:


df.head()


# In[12]:


array=df.values
X=array[:,0:8]
y=array[:,8]
seed=7
print(df.shape)
print(X.shape)
print(y.shape)


# In[35]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


# In[37]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[38]:


kfold = KFold(n_splits=10,random_state=7)


# In[41]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
result = model.score(X_test,y_test)
print("Accuracy with partitioning:{}".format(result))


# In[42]:


model2 = LogisticRegression()
results = cross_val_score(model2,X,y,cv=kfold)
print("Accuracy with CV:{}".format(results.mean()))

