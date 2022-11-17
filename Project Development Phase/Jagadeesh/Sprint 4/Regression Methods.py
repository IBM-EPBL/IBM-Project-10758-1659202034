#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[113]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[114]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='NgS5Cy5_ZLQF6mBGDOhVIA7GfRn5bRxmnryffm-IuADk',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'universityadmitpredictor-donotdelete-pr-pkdavbjvhsmouo'
object_key = 'Admission_Predict.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# # Analyse the data

# In[115]:


df.head()


# In[116]:


df.tail()


# # Drop Serial No. Column

# In[117]:


df.drop("Serial No.",axis=1,inplace=True)


# In[118]:


df.head()


# # Checking for Null values

# In[119]:


df.isnull().sum()


# # Getting Information about dataframe

# In[120]:


df.info()


# # Statistical Summary of Dataframe

# In[121]:


df.describe()


# # To find correlation of columns

# In[122]:


corr_matrix=df.corr()


# In[123]:


corr_matrix


# # Correlation matrix as a heatmap

# In[124]:


fig = plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,annot=True)
plt.show()


# # Data Visualization

# Univariant Analysis

# In[125]:


sns.distplot(df["GRE Score"])


# In[126]:


sns.distplot(df["TOEFL Score"])


# In[127]:


sns.distplot(df["University Rating"])


# In[128]:


sns.distplot(df["SOP"])


# In[129]:


sns.distplot(df["LOR "])


# In[130]:


sns.distplot(df["CGPA"])


# In[131]:


sns.distplot(df["Research"])


# In[132]:


sns.distplot(df["Chance of Admit "])


# Bivariant Analysis

# In[133]:


sns.relplot(data=df,x="GRE Score",y="Chance of Admit ",hue="Research")
plt.title("GRE Score vs Chance of Admit")
plt.show()


# In[134]:


sns.relplot(data=df,x="TOEFL Score",y="Chance of Admit ",hue="Research",kind="line",ci=None)
plt.title("TOEFL vs Chance of Admit")
plt.show()


# In[135]:


sns.relplot(data=df,x="CGPA",y="Chance of Admit ",hue="Research")
plt.title("GRE Score vs Chance of Admit")
plt.show()


# In[136]:


sns.relplot(data=df,x="SOP",y="Chance of Admit ",hue="Research",kind="line",ci=None)
plt.title("GRE Score vs Chance of Admit")
plt.show()


# In[137]:


sns.relplot(data=df,x="LOR ",y="Chance of Admit ",hue="Research",kind="line",ci=None)
plt.title("GRE Score vs Chance of Admit")
plt.show()


# In[138]:


sns.barplot(data=df,x="University Rating",y="Chance of Admit ")
plt.title("University Rating vs Chance of Admit")
plt.show()


# In[139]:


df.hist(bins = 30, figsize = (20,20), color = 'blue')


# # Importing the required Libraries for regression model

# In[140]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# # Splitting dataset into dependent and independent columns

# In[141]:


x = df[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA"]]
y = df["Chance of Admit "]


# In[142]:


x.head()


# In[143]:


y.head()


# # Splitting dataset into training and testing data

# In[144]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# # Multiple Linear Regression

# In[145]:


multiple_lin_reg = LinearRegression()
multiple_lin_reg.fit(x_train,y_train)

y_pred_mlr = multiple_lin_reg.predict(x_test)

r2_score_mlr = r2_score(y_test,y_pred_mlr)
print("Mutiple Linear Regression's Score = {:.3f}".format(r2_score_mlr))


# # Random forest regression

# In[146]:


ran_for_reg = RandomForestRegressor(n_estimators=100,random_state=1)
ran_for_reg.fit(x_train,y_train)

y_pred_rfr = ran_for_reg.predict(x_test)

r2_score_rfr = r2_score(y_test,y_pred_rfr)
print("Random Forest Regression's Score = {:.3f}".format(r2_score_rfr))


# # Save the model

# In[147]:


import pickle


# In[148]:


pickle.dump(multiple_lin_reg,open("model.pkl","wb"))


# In[149]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[150]:


from ibm_watson_machine_learning import APIClient


# In[151]:


wml_credentails={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"13S6-gvuJHw0EgY7HAmtl8ae5tQlGcbahHYBYAacEOQn"
}
client=APIClient(wml_credentails)


# In[152]:


def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"]==space_name)['metadata']['id'])


# In[153]:


space_uid=guid_from_space_name(client,'models')
print("Space UID = "+space_uid)


# In[154]:


client.set.default_space(space_uid)


# In[155]:


client.software_specifications.list()


# In[156]:


software_spec_uid=client.software_specifications.get_id_by_name('runtime-22.1-py3.9')
software_spec_uid


# In[158]:


model_details=client.repository.store_model(model=multiple_lin_reg,meta_props={
    client.repository.ModelMetaNames.NAME:"Multiple_Linear_Regression",
    client.repository.ModelMetaNames.TYPE:'scikit-learn_1.0',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid,
},
training_data=x_train,
training_target=y_train
               )


# In[159]:


model_details


# In[ ]:




