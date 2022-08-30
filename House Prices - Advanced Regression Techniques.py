#!/usr/bin/env python
# coding: utf-8

# # House Prices - Advanced Regression Techniques
# 
# 
# ### 1. EDA (Univariate, Multivariate, KDE, Pearson Correlation)
# 
# ### 2. Data Preprocessing (Imputation, create at least 2 new features)
# ### 3. Cross-validation
# ### 4. An inference pipeline consisting of Data Preprocessing and prediction.
# 
# 
# ![house.png](attachment:house.png)

# ## IMPORTING REQUIRED LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## LOADING THE DATA SETS

# In[2]:


house_train=pd.read_csv("D:\\data set\\house-prices-advanced-regression-techniques\\train.csv")
house_test= pd.read_csv("D:\\data set\\house-prices-advanced-regression-techniques\\test.csv")


# ## Exploratory Data Analysis (EDA) 

# In[3]:


house_train.head() # to display top 5 rows


# In[4]:


house_train.tail()  # to display last 5 rows


# In[5]:


house_train.info() # to get the information on the different features, its data types and no of non null values.


# FROM ABOVE WE CAN OBSERVE THAT 
# OUR DATA SET CONTAINS "43" OBJECT DATA TYPE, "38" NUMERICAL DATA TYPE ("3" FLOAT, "35" INT)

# In[6]:


house_train.isna().sum().sort_values(ascending=False)


# from above we can observe that column as "PoolQ","MiscFeature","Alley","Fence" and "FireplaceQu" have max no of missing values

# In[7]:


house_train.describe() # statistical describtion on numerical columns


# FROM THE ABOVE CODE WE CAN FIND THE MIN VALUE AND THE MAX VALUE OF THE RESPECTIVE COLUMN IE THE SPREAD OF DATA.
# WE CAN OBSERVE THAT IN MANY COLUMNS MEAN IS GREATER THAN MEDIAN SO OUR DATA HAVE POSITIVE/RIGHT SKEW.

# In[8]:


house_train.select_dtypes(include=['int64', 'float64']).columns


# In[9]:


house_train.select_dtypes(include=['object']).columns


# ## Univariate Analysis (EDA)

# In[10]:


col=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition','MSSubClass','OverallQual','OverallCond','GarageCars']


# In[12]:


plt.figure(figsize=(22,300))
for i in range(len(col)):
  plt.subplot(100,3,i+1)
  sns.countplot(house_train[col[i]])
  plt.title(f"Countplot of {col[i]}",fontsize=18)
  plt.xticks(rotation=90,fontsize=15)
  plt.tight_layout()


# ## Observation
# 
# From the above graphs we can observe that
# a) In MSZoning people like to stay where there is low population.
# b) people like to have their house in Pave street type.
# c) from lotshape people like Reg and dont like IR2,IR3.etc
# 

# In[13]:


house_train.hist(figsize=(22,30))


# In[14]:


data=house_train.select_dtypes(exclude='object')


# In[15]:


X_col=data.columns.values


# ## Checking outliner

# In[16]:


plt.figure(figsize=(14,30))
for i in range(0,len(X_col)):
  plt.subplot(20,5,i+1)
  ax=sns.boxplot(data[X_col[i]],color='blue')
  plt.tight_layout()


# There are many outliners we can observe from the above box plot.

# ### Treating missing values

# In[17]:


# Show the null values using heatmap
plt.figure(figsize=(16,9))
sns.heatmap(house_train.isnull())


# In[18]:


# Get the percentages of null value
null_percent = house_train.isnull().sum()/house_train.shape[0]*100
null_percent


# In[20]:


col_for_drop = null_percent[null_percent > 20].keys() # if the null value % 20 or > 20 so need to drop it
col_for_drop


# In[22]:


# drop columns
df = house_train.drop(col_for_drop, "columns")
df.shape


# In[23]:


df.columns


# # Data Processing

# ## Fill Missing Values

# In[24]:



df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[25]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[26]:


df.shape


# In[27]:


df.drop(['Id'],axis=1,inplace=True)
df.isnull().sum()


# In[28]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[29]:


df.dropna(inplace=True)


# In[30]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[31]:


df.shape


# In[32]:


# correlation heatmap
plt.figure(figsize=(25,25))
ax = sns.heatmap(df.corr(), cmap = "coolwarm", annot=True, linewidth=2)

# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# ### Correlation

# In[34]:


corr=df.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]


# from the above table we can find that some columns like "OverallQual ", " GrLivArea ", " GrLivArea " etc have correlation more than 0.5 with sales price.
# We can make use of such features to predict our target value ie sales price

# In[37]:


#plotting correlations
num_feat=df.columns[df.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df.SalePrice.values)[0,1])
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sale Price");


# we can find that some columns like "overallcod","BsmtfinSF2",etc have negative correlation with respect to salesprice.

# In[41]:


# MasVnrArea Vs SalePrice


plt.scatter(df["MasVnrArea"],df["SalePrice"])
plt.title("MasVnrArea Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Mas Vnr Area in sq feet");


# In[43]:


sns.boxplot("MasVnrType","SalePrice",data=df);


# In[44]:


df["MasVnrType"] = df["MasVnrType"].fillna('None')
df["MasVnrArea"] = df["MasVnrArea"].fillna(0.0)


# In[45]:


sns.boxplot("MasVnrType","SalePrice",data=df);


# In[46]:


fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[48]:


#Deleting outliers
train = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# # TARGET VALUE

# In[54]:


sns.distplot(df['SalePrice']);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# In[ ]:




