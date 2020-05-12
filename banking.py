#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import pickle
import numpy as np


# In[29]:


data=pd.read_csv("d:b.csv",delimiter=',')


# In[30]:


data


# In[ ]:





# In[31]:


data


# In[32]:


# DATA CLEANING


# In[33]:


# DATA CLEANING PART-1
# REMOVING THE (-)SPECIAL CHARCTER FROM JOB COLUMN AND REPLACEING THROUGH SPACE..
data['job']=data['job'].str.replace('-',' ')
# REMOVING THE (.)CHARACTER  FROM JOB COLUMN AND REPLACEING THROUG NOTHING..
data['job']=data['job'].str.replace('.','')


# In[34]:


# DATA CLEANING PART-2
# REMOVING THE (.4Y),(.6Y),(.9Y) FROM THE COLUMN EDUCATION
data['education']=data['education'].str.replace(".4y","")
data['education']=data['education'].str.replace(".6y","")
data['education']=data['education'].str.replace(".9y","")
# REMOVING (.) BETWEEN THE EDUCTION FIELD AND REPLACING THROUGH SPACE
data['education']=data["education"].str.replace(".",' ')





# DATA CLEANING HAS DONE 




# In[ ]:





# In[35]:


# DATA CLEANING PART-3
# FILLING UNKNOWN VALUE WITH NONE IN COLUMN EDUCATION
data['education']=data['education'].str.replace("unknown","None")
# FILLING UNKNOWN VALUE WITH no(because of mean) IN COLUMN default
data['default']=data['default'].str.replace("unknown","No")
# FILLING UNKNOWN VALUE WITH no(because of mean) IN COLUMN housing
data['housing']=data['housing'].str.replace("unknown","No")
# FILLING UNKNOWN VALUE WITH no(because of mean) IN COLUMN loan
data['loan']=data['loan'].str.replace("unknown","No")


# In[36]:


# DATA PREPROCESSING


# In[37]:


# DATA PREPROCESSING PART-1
from sklearn.preprocessing import LabelEncoder
m=LabelEncoder()
# converting the job categorial column into integer format
data["job"]=m.fit_transform(data["job"])
# converting the marital categorial column into integer format
data["marital"]=m.fit_transform(data["marital"])
# converting the education categorial column into integer format
data["education"]=m.fit_transform(data["education"])
# converting the default categorial column into integer format
data["default"]=m.fit_transform(data["default"])
# converting the housing categorial column into integer format
data["housing"]=m.fit_transform(data["housing"])
# converting the loan categorial column into integer format
data["loan"]=m.fit_transform(data["loan"])
#converting the contact categorial column into integer format
data["contact"]=m.fit_transform(data["contact"])
#converting the month categorial column into integer format
data["month"]=m.fit_transform(data["month"])
#converting the day_of_week categorial column into integer format
data["day_of_week"]=m.fit_transform(data["day_of_week"])
#converting the poutcome categorial column into integer format
data["poutcome"]=m.fit_transform(data["poutcome"])
#converting the y(targeted_column) categorial column into integer format
data["y"]=m.fit_transform(data["y"])


# Preprocessing has done














# In[38]:


data


# In[39]:


# splitting the data


# In[40]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=87)


# In[42]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
l=SelectFromModel(Lasso(alpha=0.005))


# In[43]:


l.fit(x_train,y_train)


# In[44]:


# the features which is not selected
l.get_support()


# In[45]:


# name of features which got selected
selected_features=x_train.columns[(l.get_support())]
selected_features


# In[ ]:







# In[46]:


# NW IMPORTING ALGORITHIUM
from sklearn.ensemble import RandomForestClassifier
p=RandomForestClassifier()
p.fit(x_train,y_train)


# In[47]:


o=p.predict(x_test)


# In[48]:


# from matrix... to know type1 error and type 2 error
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,o)


# In[49]:


# To know the accuracy of our model
from sklearn.metrics import accuracy_score
accuracy_score(y_test,o)


# In[50]:


pickle.dump(p, open('Banking.pkl','wb'))


# In[ ]:





# In[63]:


z=[["Student"]]


# In[64]:



m.transform(z)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




