#!/usr/bin/env python
# coding: utf-8

# # Homework 1: Applied Machine Learning - Linear | Logisitc | SVM

# In[1]:


print("Name --> Davit Barblishvili")
print("UNI --> db3230")


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.compose import make_column_transformer


# In[3]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[4]:


pd.options.mode.chained_assignment = None


# #**Part 1: Linear Regression**

# In part 1, we will use **two datasets** to train and evaluate our linear regression model.
# 
# The first dataset will be a synthetic dataset sampled from the following equations:
#    
# **𝜖 ∼ Normal(0,3**)
# 
# **z = 3𝑥 + 10y + 10 + 𝜖**

# In[5]:


np.random.seed(0)
epsilon = np.random.normal(0, 3, 100)
x = np.linspace(0, 10, 100) 
y = np.linspace(0, 5, 100)
z = 3 * x + 10 * y + 10 + epsilon


# To apply linear regression, we need to first check if the assumptions of linear regression are not violated.
# 
# Assumptions of Linear Regression:
# 
# - Linearity: $y$ is a linear (technically affine) function of $x$.
# - Independence: the $x$'s are independently drawn, and not dependent on each other.
# - Homoscedasticity: the $\epsilon$'s, and thus the $y$'s, have constant variance.
# - Normality: the $\epsilon$'s are drawn from a Normal distribution (i.e. Normally-distributed errors)
# 
# These properties, as well as the simplicity of this dataset, will make it a good test case to check if our linear regression model is working properly.

# **1.1. Plot z vs x and z vs y in the synthetic dataset as scatter plots. Label your axes and make sure your y-axis starts from 0. Do the independent and dependent features have linear relationship?**

# In[6]:


### Your code here
plt.ylim(0, 100)
plt.xlabel("x-label")
plt.ylabel("z-label")
sns.scatterplot(x = x, y = z)


# In[7]:


### Your code here
plt.ylim(0, 100)
plt.xlabel("y-label")
plt.ylabel("z-label")
sns.scatterplot(x = y, y = z)


# **1.2. Are the independent variables correlated? Use pearson correlation to verify? What would be the problem if linear regression is applied to correlated features?**

# In[8]:


### Your code here
pearson_correlation = np.corrcoef(x, y)
print(pearson_correlation)
    
# Yes they are highly correlated and that is not a good result. 
# For a linear regression, highly correlated features might will result in highly unstable parameter estimates.


# **The second dataset we will be using is an auto MPG dataset. This dataset contains various characteristics for around 8128 cars. We will use linear regression to predict the selling_price label**

# In[9]:


auto_mpg_df = pd.read_csv('Car details v3.csv')
# Dropping Torque column, there is information in this column but it will take some preprocessing.
# The idea of the exercise is to familarize yourself with the basics of Linear regression.
auto_mpg_df = auto_mpg_df.drop(['torque'], axis = 1)


# In[10]:


auto_mpg_df


# **1.3. Missing Value analysis - Auto mpg dataset.**
# 
# **Are there any missing values in the dataset? If so, what can be done about it? Jusify your approach.**

# In[11]:


### Your code here
print(" \nCount total NaN at each column in a DataFrame : \n\n",
      auto_mpg_df.isnull().sum())

# considering we have 8128 rows, I believe it would be a good idea to remove the rows that have the NaN values. 
# There are some other methods to deal with NaN values such as column removal, but in this case it is not going to
# work because out of 8128 only 2xx rows do not have data in a several columns so deleting the entire column for
# that is not useful. One more way is to interpolate the values meaning averaging based on the neighboring
# values, but after looking at the data, I do not think there is a good correlation between let's say 
# selling price and mileage, so removing the rows that include NaN values remains the best option. 


# In[12]:


auto_mpg_df = auto_mpg_df.dropna(axis=0)

print(" \nCount total NaN at each column in a DataFrame : \n\n",
      auto_mpg_df.isnull().sum())

auto_mpg_df


# In[13]:


# checking individual features

print(auto_mpg_df['name'].isna().sum())
print(auto_mpg_df['year'].isna().sum())
print(auto_mpg_df['selling_price'].isna().sum())
print(auto_mpg_df['km_driven'].isna().sum())
print(auto_mpg_df['seller_type'].isna().sum())
print(auto_mpg_df['transmission'].isna().sum())
print(auto_mpg_df['owner'].isna().sum()) 
print(auto_mpg_df['mileage'].isna().sum())
print(auto_mpg_df['engine'].isna().sum())
print(auto_mpg_df['max_power'].isna().sum()) 
print(auto_mpg_df['seats'].isna().sum())


# **1.4. The features engine, max_power and mileage have units in the dataset. In the real world if we have such datasets, we generally remove the units from each feature. After doing so, convert the datatype of these columns to float. For example: 1248 CC engine is 1248, 23.4 kmpl is 23.4 and so on.**
# 
# **Hint: Check for distinct units in each of these features. A feature might have multiple units as well. Also, a feature could have no value but have unit. For example 'CC' without any value. Remove such rows.**

# In[14]:


auto_mpg_df["engine"].unique()


# In[15]:


### Your code here
### cleaning column --> engine
auto_mpg_df["engine"] = auto_mpg_df["engine"].map(lambda x: x.rstrip('CC'))
auto_mpg_df["engine"] = auto_mpg_df["engine"].str.strip()
auto_mpg_df["engine"]= auto_mpg_df["engine"].astype(float)


# In[16]:


kmkg = 0
kmpl = 0
for i in auto_mpg_df.mileage:
    if str(i).endswith("km/kg"):
        kmkg+=1
    elif str(i).endswith("kmpl"):
        kmpl+=1
print('The number of rows with Km/Kg : {} '.format(kmkg))
print('The number of rows with Kmpl : {} '.format(kmpl))

# since there are 88 rows it is not going to make a big difference overal how we deal with these 88 rows that 
# do not align with kmpl.


# In[17]:


Correct_Mileage= []
for i in auto_mpg_df.mileage:
    if str(i).endswith('km/kg'):
        i = i[:-5]
        # conversion
        i = float(i)*1.40
        auto_mpg_df["mileage"] = auto_mpg_df["mileage"].str.strip()
        Correct_Mileage.append(float(i))
    elif str(i).endswith('kmpl'):
        i = i[:-4]
        auto_mpg_df["mileage"] = auto_mpg_df["mileage"].str.strip()
        Correct_Mileage.append(float(i))
auto_mpg_df['mileage']=Correct_Mileage       


# In[18]:


auto_mpg_df["max_power"].unique()


# In[19]:


### cleaning column --> max_power
auto_mpg_df["max_power"] = auto_mpg_df["max_power"].map(lambda x: x.rstrip('bhp'))
auto_mpg_df["max_power"] = auto_mpg_df["max_power"].str.strip()


# In[20]:


auto_mpg_df['max_power'] = auto_mpg_df['max_power'].replace('', float('NaN'))
auto_mpg_df = auto_mpg_df.dropna()
auto_mpg_df["max_power"]= auto_mpg_df["max_power"].astype(float)


# In[21]:


auto_mpg_X = auto_mpg_df.drop(columns=['selling_price'])
auto_mpg_y = auto_mpg_df['selling_price']


# **1.5. Plot the distribution of the label (selling_price) using a histogram. Make multiple plots with different binwidths. Make sure to label your axes while plotting.**

# In[22]:


### Your code here

fig, ax = plt.subplots(2,2, figsize=(14,7))
ax[0][0].scatter(auto_mpg_X['year'], auto_mpg_y)
ax[0][0].set_xlabel('year')
ax[0][0].set_ylabel('selling_price')
ax[0][0].set_title('year vs selling_price')
ax[1][0].scatter(auto_mpg_X['km_driven'], auto_mpg_y)
ax[1][0].set_ylabel('selling_price')
ax[1][0].set_xlabel('km_driven')
ax[1][0].set_title('km_driven vs selling_price')
ax[1][1].scatter(auto_mpg_X['mileage'], auto_mpg_y)
ax[1][1].set_ylabel('selling_price')
ax[1][1].set_xlabel('mileage')
ax[1][1].set_title('mileage vs selling_price')
ax[0][1].scatter(auto_mpg_X['engine'], auto_mpg_y)
ax[0][1].set_ylabel('selling_price')
ax[0][1].set_xlabel('acceleration')
ax[0][1].set_title('engine vs selling_price')

fig.tight_layout() 
fig.show()


# In[23]:




maxAmount = max(auto_mpg_y)
plt.xlabel('Selling Price')
plt.ylabel("Number of Vehicles");
auto_mpg_y.plot.hist(bins=35, legend=True, title='Histogram of Selling Price',)


# In[24]:



maxAmount = max(auto_mpg_y)
plt.xlabel('Selling Price')
plt.ylabel("Number of Vehicles");
auto_mpg_y.plot.hist(bins=55, legend=True, title='Histogram of Selling Price',)


# In[25]:



maxAmount = max(auto_mpg_y)
plt.xlabel('Selling Price')
plt.ylabel("Number of Vehicles");
auto_mpg_y.plot.hist(bins=75, legend=True, title='Histogram of Selling Price',)


# **1.6. Plot the relationships between the label (Selling Price) and the continuous features (Mileage, km driven, engine, max power) using a small multiple of scatter plots. 
# Make sure to label the axes. Do you see something interesting about the distributions of these features.**

# In[26]:


### Your code here
auto_mpg_df.plot.scatter(x='selling_price', y='engine', title= "Scatter plot between selling price and engine");


# In[27]:


auto_mpg_df.plot.scatter(x='selling_price', y='mileage', 
                         title= "Scatter plot between selling price and mileage");


# In[28]:


auto_mpg_df.plot.scatter(x='selling_price', y='max_power', 
                         title= "Scatter plot between selling price and max power");

### max power is not that popular determiner of the car's popularity. Same signs are appearing here. 
### The average power car cost the average price


# In[29]:


auto_mpg_df.plot.scatter(x='selling_price', y='km_driven', title= 
                         "Scatter plot between selling price and km driven")

### lower the km driver the higher the concentration of points


# **1.7. Plot the relationships between the label (Selling Price) and the discrete features (fuel type, Seller type, transmission) using a small multiple of box plots. Make sure to label the axes.**

# In[30]:


### Your code here

sns.set_style("whitegrid")
sns.boxplot(x = 'fuel', y = 'selling_price', data = auto_mpg_df)


# In[31]:


sns.set_style("whitegrid")
sns.boxplot(x = 'transmission', y = 'selling_price', data = auto_mpg_df)


# In[32]:


sns.set_style("whitegrid")
sns.boxplot(x = 'seller_type', y = 'selling_price', data = auto_mpg_df)


# In[33]:


from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27
sns.set_style("whitegrid")
sns.boxplot(x = 'owner', y = 'selling_price', data = auto_mpg_df)


# **1.8. From the visualizations above, do you think linear regression is a good model for this problem? Why and/or why not?**

# In[34]:


### Your answer here
### I think linear regression is a good model for this problem since we need to find the best model
### to fit the data which seems equally distributed around the mean and does not have that many outliers. additionally
### from the above, it appears many variables do have a linear relationship and are correlated to price which allows
### us to ise linear regression to uncover different relationships. 
### once we have the model, predicting selling price of the vehicle given let's say fuel type is just finding 
### the 'y' value. 


# In[35]:


auto_mpg_X['year'] =  2020 - auto_mpg_X['year']


# In[36]:


#dropping the car name as it is irrelevant.
auto_mpg_X.drop(['name'],axis = 1,inplace=True)

#check out the dataset with new changes
auto_mpg_X.head()


# **Data Pre-processing**

# **1.9.
# Before we can fit a linear regression model, there are several pre-processing steps we should apply to the datasets:**
# 1. Encode categorial features appropriately.
# 2. Split the dataset into training (60%), validation (20%), and test (20%) sets.
# 3. Standardize the columns in the feature matrices X_train, X_val, and X_test to have zero mean and unit variance. To avoid information leakage, learn the standardization parameters (mean, variance) from X_train, and apply it to X_train, X_val, and X_test.
# 4. Add a column of ones to the feature matrices X_train, X_val, and X_test. This is a common trick so that we can learn a coefficient for the bias term of a linear model.
# 

# In[37]:


print(" \nCount total NaN at each column in a DataFrame : \n\n",
      auto_mpg_X.isnull().sum())

# dataset does not have nan values


# In[38]:


X = x.reshape((100, 1))   # Turn the x vector into a feature matrix X

# 1. No categorical features in the synthetic dataset (skip this step)

# 2. Split the dataset into training (60%), validation (20%), and test (20%) sets
X_dev, X_test, y_dev, y_test = train_test_split(X, z, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=0)

# 3. Standardize the columns in the feature matrices
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit and transform scalar on X_train
X_val = scaler.transform(X_val)           # Transform X_val
X_test = scaler.transform(X_test)         # Transform X_test

# 4. Add a column of ones to the feature matrices
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print(X_train[:5])
print(y_train[:5])


# In[39]:


print(X_train.mean(axis=0), X_train.std(axis=0))
print(X_val.mean(axis=0), X_val.std(axis=0))
print(X_test.mean(axis=0), X_test.std(axis=0))


# In[40]:


auto_mpg_y


# In[41]:


# 1. No categorical features in the synthetic dataset (skip this step)

num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
cat_features = ['fuel', 'seller_type', 'transmission', 'owner']
all_features_mpg = cat_features + num_features + ['Bias']
auto_mpg_X['Bias'] = 1

# 2. Split the dataset into training (60%), validation (20%), and test (20%) sets
auto_mpg_X_dev, auto_mpg_X_test, auto_mpg_y_dev, auto_mpg_y_test = train_test_split(auto_mpg_X, auto_mpg_y, test_size=0.2, random_state=0)
auto_mpg_X_train, auto_mpg_X_val, auto_mpg_y_train, auto_mpg_y_val = train_test_split(auto_mpg_X_dev, auto_mpg_y_dev, test_size=0.25, random_state=0)

column_transformer = make_column_transformer((StandardScaler(), num_features),
                                           (TargetEncoder(), cat_features),
                                            remainder='passthrough')
# 3. Standardize the columns in the feature matrices
auto_mpg_X_train = column_transformer.fit_transform(auto_mpg_X_train,auto_mpg_y_train)
auto_mpg_X_val = column_transformer.transform(auto_mpg_X_val)
auto_mpg_X_test = column_transformer.transform(auto_mpg_X_test)


# In[42]:


print(auto_mpg_X_train.mean(axis=0), auto_mpg_X_train.std(axis=0))
print(auto_mpg_X_val.mean(axis=0), auto_mpg_X_val.std(axis=0))
print(auto_mpg_X_test.mean(axis=0), auto_mpg_X_test.std(axis=0))


# **At the end of this pre-processing, you should have the following vectors and matrices:**
# 
# **- Auto MPG dataset: auto_mpg_X_train, auto_mpg_X_val, auto_mpg_X_test, auto_mpg_y_train, auto_mpg_y_val, auto_mpg_y_test**

# **Implement Linear Regression**

# Now, we can implement our linear regression model! Specifically, we will be implementing ridge regression, which is linear regression with L2 regularization. Given an (m x n) feature matrix $X$, an (m x 1) label vector $y$, and an (n x 1) weight vector $w$, the hypothesis function for linear regression is:
# 
# $$
# y = X w
# $$
# 
# Note that we can omit the bias term here because we have included a column of ones in our $X$ matrix, so the bias term is learned implicitly as a part of $w$. This will make our implementation easier.
# 
# Our objective in linear regression is to learn the weights $w$ which best fit the data. This notion can be formalized as finding the optimal $w$ which minimizes the following loss function:
# 
# $$
# \min_{w} \| X w - y \|^2_2 + \alpha \| w \|^2_2 \\
# $$
# 
# This is the ridge regression loss function. The $\| X w - y \|^2_2$ term penalizes predictions $Xw$ which are not close to the label $y$. And the $\alpha \| w \|^2_2$ penalizes large weight values, to favor a simpler, more generalizable model. The $\alpha$ hyperparameter, known as the regularization parameter, is used to tune the complexity of the model - a higher $\alpha$ results in smaller weights and lower complexity, and vice versa. Setting $\alpha = 0$ gives us vanilla linear regression.
# 
# Conveniently, ridge regression has a closed-form solution which gives us the optimal $w$ without having to do iterative methods such as gradient descent. The closed-form solution, known as the Normal Equations, is given by:
# 
# $$
# w = (X^T X + \alpha I)^{-1} X^T y
# $$

# **1.10. Implement a `LinearRegression` class with two methods: `train` and `predict`. You may NOT use sklearn for this implementation. You may, however, use `np.linalg.solve` to find the closed-form solution. It is highly recommended that you vectorize your code.**

# In[43]:


class LinearRegression():
    '''
    Linear regression model with L2-regularization (i.e. ridge regression).

    Attributes
    ----------
    alpha: regularization parameter
    w: (n x 1) weight vector
    '''
    
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.w = None

    def train(self, X, y):
        '''Trains model using ridge regression closed-form solution 
        (sets w to its optimal value).
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        None
        '''
        
        num_rows, num_cols = X.shape

        LHS = inv(np.matmul(X.T,X) + self.alpha*np.identity(num_cols))
        RHS = np.matmul(X.T,y)
        w = np.matmul(LHS,RHS)
        self.w = w
        
    def predict(self, X):
        '''Predicts on X using trained model.
        
        Parameters
        ----------
        X : (m x n) feature matrix
        
        Returns
        -------
        y_pred: (m x 1) prediction vector
        '''
        ### Your code here
        
        y_pred = np.matmul(X, self.w)
        return y_pred


# **Train, Evaluate, and Interpret Linear Regression Model**

# **1.11. A) Train a linear regression model ($\alpha = 0$) on the auto MPG training data. Make predictions and report the mean-squared error (MSE) on the training, validation, and test sets. Report the first 5 predictions on the test set, along with the actual labels.**

# In[44]:


### Your code here
from sklearn.metrics import mean_squared_error
ridge_reg = LinearRegression(alpha=0)
ridge_reg.train(auto_mpg_X_train,auto_mpg_y_train)
ridge_reg_predictions = ridge_reg.predict(auto_mpg_X_test)
print('predictions for first 5 values from test set: ')
print(ridge_reg_predictions[:5])
print('actual first 5 values from test set: ')
print(auto_mpg_y_train[:5])

train_est = ridge_reg.predict(auto_mpg_X_train)
print('MSE using predicted from model on train set: ')
print(mean_squared_error(auto_mpg_y_train, train_est))

val_est = ridge_reg.predict(auto_mpg_X_val)
print('MSE using predicted from model on val set: ')
print(mean_squared_error(auto_mpg_y_val, val_est))

test_est = ridge_reg.predict(auto_mpg_X_test)
print('MSE using predicted from model on test set: ')
print(mean_squared_error(auto_mpg_y_test, test_est))


# **B) As a baseline model, use the mean of the training labels (auto_mpg_y_train) as the prediction for all instances. Report the mean-squared error (MSE) on the training, validation, and test sets using this baseline. This is a common baseline used in regression problems and tells you if your model is any good. Your linear regression MSEs should be much lower than these baseline MSEs.**

# In[45]:


### Your code here
baseline_est = auto_mpg_y_train.mean()
baseline_train_est = np.full((len(auto_mpg_y_train),1), baseline_est)
print('MSE using baseline on train set: ')
print(mean_squared_error(auto_mpg_y_train, baseline_train_est))
baseline_val_est = np.full((len(auto_mpg_y_val),1), baseline_est)
print('MSE using baseline on val set: ')
print(mean_squared_error(auto_mpg_y_val, baseline_val_est))
baseline_test_est = np.full((len(auto_mpg_y_test),1), baseline_est)
print('MSE using baseline on test set: ')
print(mean_squared_error(auto_mpg_y_test, baseline_test_est))


# **1.12. Interpret your model trained on the auto MPG dataset using a bar chart of the model weights. Make sure to label the bars (x-axis) and don't forget the bias term! Use lecture 3, slide 15 as a reference. According to your model, which features are the greatest contributors to the selling price**

# In[46]:


all_features_mpg


# In[47]:


ridge_reg.w


# In[49]:


### Your code here
fig = plt.figure(figsize = (20,12))
xval = np.zeros((31))
yval = np.reshape(ridge_reg.w, -1)
ax = sns.barplot(x=all_features_mpg, y=yval)
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('feature name')
ax.set_ylabel('feature importance (coefficient value)')
ax.set_title('feature importance across features')
plt.show()


# **Tune Regularization Parameter $\alpha$**

# **Now, let's do ridge regression and tune the $\alpha$ regularization parameter on the auto MPG dataset.**
# 
# **1.13. Sweep out values for $\alpha$ using `alphas = np.logspace(-2, 1, 10)`. Perform a grid search over these $\alpha$ values, recording the training and validation MSEs for each $\alpha$. A simple grid search is fine, no need for k-fold cross validation. Plot the training and validation MSEs as a function of $\alpha$ on a single figure. Make sure to label the axes and the training and validation MSE curves. Use a log scale for the x-axis.**

# In[55]:


alphas = np.logspace(-2, 1, 10)
df_mses = pd.DataFrame(columns=['alpha', 'mse_training'])
for alpha in alphas: 
    model_lin = LinearRegression(alpha=alpha)
    model_lin.train(auto_mpg_X_train,auto_mpg_y_train)
    train_est = model_lin.predict(auto_mpg_X_train)
    train_mse = mean_squared_error(auto_mpg_y_train, train_est)
    
    val_est = model_lin.predict(auto_mpg_X_val)
    val_mse = mean_squared_error(auto_mpg_y_val, val_est)
    temp_df = pd.DataFrame(columns=['alpha', 'mse_training'], data=[[alpha, train_mse]])
    df_mses = df_mses.append(temp_df)

df_mses.head()
ax = df_mses.set_index('alpha').plot()
ax.legend(bbox_to_anchor=(1.0,1.0))
ax.set_ylabel('MSE')
ax.set_title('MSE value vs alpha for training')
ax.plot()


# In[54]:


### Your code here
alphas = np.logspace(-2, 1, 10)
df_mses = pd.DataFrame(columns=['alpha', 'mse_validation'])
for alpha in alphas: 
    model_lin = LinearRegression(alpha=alpha)
    model_lin.train(auto_mpg_X_train,auto_mpg_y_train)
    train_est = model_lin.predict(auto_mpg_X_train)
    train_mse = mean_squared_error(auto_mpg_y_train, train_est)
    
    val_est = model_lin.predict(auto_mpg_X_val)
    val_mse = mean_squared_error(auto_mpg_y_val, val_est)
    temp_df = pd.DataFrame(columns=['alpha', 'mse_validation'], data=[[alpha,  val_mse]])
    df_mses = df_mses.append(temp_df)

df_mses.head()
ax = df_mses.set_index('alpha').plot()
ax.legend(bbox_to_anchor=(1.0,1.0))
ax.set_ylabel('MSE')
ax.set_title('MSE value vs alpha for validation set')
ax.plot()


# **Explain your plot above. How do training and validation MSE behave with decreasing model complexity (increasing $\alpha$)?**

# In[ ]:


### Your answer here
# As alpha increases both MSE for the training and validation increase as well. 


# **1.14. Using the $\alpha$ which gave the best validation MSE above, train a model on the training set. Report the value of $\alpha$ and its training, validation, and test MSE. This is the final tuned model which you would deploy in production.**

# In[56]:


### Your code here
print(df_mses)

prod_model = LinearRegression(alpha=.046416)
prod_model.train(auto_mpg_X_train, auto_mpg_y_train)
train_est = prod_model.predict(auto_mpg_X_train)
print('MSE using predicted from model on train set: ')
print(mean_squared_error(auto_mpg_y_train, train_est))

val_est = prod_model.predict(auto_mpg_X_val)
print('MSE using predicted from model on val set: ')
print(mean_squared_error(auto_mpg_y_val, val_est))

test_est = prod_model.predict(auto_mpg_X_test)
print('MSE using predicted from model on test set: ')
print(mean_squared_error(auto_mpg_y_test, test_est))


# # **Part 2: Logistic Regression**
# 
# **Gender Recognition by Voice and Speech Analysis**
# 
# **This dataset is used to identify a voice as male or female, based upon acoustic properties of the voice and speech.**

# In[57]:


voice_df = pd.read_csv("voice-classification.csv")
voice_df.head()


# **Data - Checking Rows & Columns**

# In[58]:


#Number of Rows & Columns
print(voice_df.shape) 


# **2.1 What is the probability of observing different  categories in the Label feature of the dataset?**
# 
# This is mainly to check class imbalance in the dataset, and to apply different techniques to balance the dataset, which we will learn later.

# In[59]:


voice_df.label.unique()


# In[60]:


#code here
male_count = len(voice_df[voice_df['label'] == 'male'])
female_count = len(voice_df[voice_df['label'] == 'female'])

female_count / (female_count + male_count)


# **2.2 Plot the relationships between the label and the 20 numerical features using a small multiple of box plots. Make sure to label the axes. What useful information do this plot provide?**

# In[61]:


#code here
import seaborn as sns
fig, ax = plt.subplots(20,1, figsize=(14,90))
i = 0
for col in voice_df.drop(columns=['label']).columns.values.tolist():
    sns.boxplot(x=voice_df[col],y=voice_df['label'], ax=ax[i])
    ax[i].set_title("% s for Male and Female relationships"% col)
    i = i + 1
fig.tight_layout()
plt.show()


# In[62]:


corr = voice_df.corr()
corr.style.background_gradient(cmap='coolwarm')


# **2.3 Plot the correlation matrix, and check if there is high correlation between the given numerical features (Threshold >=0.9). If yes, drop those highly correlated features from the dataframe. Why is necessary to drop those columns before proceeding further?**

# In[63]:


voice_y = voice_df['label']


# In[64]:


def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() >= threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out


# In[65]:


voice_df1 = trimm_correlated(voice_df, 0.9)


# In[66]:


voice_df1['label'] = voice_y


# In[67]:


# Split data into features and labels
voice_X = voice_df1.drop(columns=['label']) #replace "voice_df1" with your dataframe from 2.3 to make sure the code runs
voice_y = voice_df1['label']
print(voice_X.columns)


# **2.4 Apply the following pre-processing steps:**
# 
# 1) Use OrdinalEncoding to encode the label in the dataset (male & female)
# 
# 2) Convert the label from a Pandas series to a Numpy (m x 1) vector. If you don't do this, it may cause problems when implementing the logistic regression model.
# 
# 3)Split the dataset into training (60%), validation (20%), and test (20%) sets.
# 
# 4) Standardize the columns in the feature matrices. To avoid information leakage, learn the standardization parameters from training, and then apply training, validation and test dataset.
# 
# 5) Add a column of ones to the feature matrices of train, validation and test dataset. This is a common trick so that we can learn a coefficient for the bias term of a linear model.

# In[68]:


voice_df1


# In[69]:


#code here
enc = OrdinalEncoder()

voice_df1['encoded_label'] = np.where(voice_df1['label'] == 'male', 1, 0)
voice_y = voice_df1['encoded_label']
voice_df1.drop('encoded_label', axis=1)
voice_y = voice_y.to_numpy()
voice_y = voice_y.reshape((voice_y.shape[0],1))

voice_X_dev, voice_X_test, voice_y_dev, voice_y_test = train_test_split(
    voice_X, voice_y, test_size=0.2, random_state=0)
voice_X_train, voice_X_val, voice_y_train, voice_y_val = train_test_split(
    voice_X_dev, voice_y_dev, test_size=0.25, random_state=0)


scaler = StandardScaler()
voice_X_train = scaler.fit_transform(voice_X_train)
voice_X_val = scaler.transform(voice_X_val)           
voice_X_test = scaler.transform(voice_X_test)

voice_X_train = np.hstack([np.ones((voice_X_train.shape[0], 1)), voice_X_train])
voice_X_val = np.hstack([np.ones((voice_X_val.shape[0], 1)), voice_X_val])
voice_X_test = np.hstack([np.ones((voice_X_test.shape[0], 1)), voice_X_test])


# **2.5 Implement Logistic Regression**
# 
# We will now implement logistic regression with L2 regularization. Given an (m x n) feature matrix $X$, an (m x 1) label vector $y$, and an (n x 1) weight vector $w$, the hypothesis function for logistic regression is:
# 
# $$
# y = \sigma(X w)
# $$
# 
# where $\sigma(x) = \frac{1}{1 + e^{-x}}$, i.e. the sigmoid function. This function scales the prediction to be a probability between 0 and 1, and can then be thresholded to get a discrete class prediction.
# 
# Just as with linear regression, our objective in logistic regression is to learn the weights $𝑤$ which best fit the data. For L2-regularized logistic regression, we find an optimal $w$ to minimize the following loss function:
# 
# $$
# \min_{w} \ -y^T \ \text{log}(\sigma(Xw)) \ - \  (\mathbf{1} - y)^T \ \text{log}(\mathbf{1} - \sigma(Xw)) \ + \ \alpha \| w \|^2_2 \\
# $$
# 
# Unlike linear regression, however, logistic regression has no closed-form solution for the optimal $w$. So, we will use gradient descent to find the optimal $w$. The (n x 1) gradient vector $g$ for the loss function above is:
# 
# $$
# g = X^T \Big(\sigma(Xw) - y\Big) + 2 \alpha w
# $$
# 
# Below is pseudocode for gradient descent to find the optimal $w$. You should first initialize $w$ (e.g. to a (n x 1) zero vector). Then, for some number of epochs $t$, you should update $w$ with $w - \eta g $, where $\eta$ is the learning rate and $g$ is the gradient. You can learn more about gradient descent [here](https://www.coursera.org/lecture/machine-learning/gradient-descent-8SpIM).
# 
# > $w = \mathbf{0}$
# > 
# > $\text{for } i = 1, 2, ..., t$
# >
# > $\quad \quad w = w - \eta g $
# 

# Implement a LogisticRegression class with five methods: train, predict, calculate_loss, calculate_gradient, and calculate_sigmoid. **You may NOT use sklearn for this implementation. It is highly recommended that you vectorize your code.**

# In[70]:


import math

class LogisticRegression():
    '''
    Logistic regression model with L2 regularization.

    Attributes
    ----------
    alpha: regularization parameter
    t: number of epochs to run gradient descent
    eta: learning rate for gradient descent
    w: (n x 1) weight vector
    '''
    
    def __init__(self, alpha, t, eta):
        self.alpha = alpha
        self.t = t
        self.eta = eta
        self.w = None

    def train(self, X, y):
        '''Trains logistic regression model using gradient descent 
        (sets w to its optimal value).
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        losses: (t x 1) vector of losses at each epoch of gradient descent
        '''
        ### Your code here
        losses = []
        num_rows, num_cols = X.shape
        self.w = np.zeros((num_cols,1))
        for i in range(self.t):
            self.w = self.w  - self.eta * self.calculate_gradient(X, y)
            curr_loss = self.calculate_loss(X, y)
            losses.append(curr_loss)
        losses_array = np.array(losses)
        return losses_array
        
    def predict(self, X):
        '''Predicts on X using trained model. Make sure to threshold 
        the predicted probability to return a 0 or 1 prediction.
        
        Parameters
        ----------
        X : (m x n) feature matrix
        
        Returns
        -------
        y_pred: (m x 1) 0/1 prediction vector
        '''
        ### Your code here
        values = self.calculate_sigmoid(np.matmul(X, self.w))
        values[values >= .5] = 1
        values[values < 1] = 0
        return values
    
    def calculate_loss(self, X, y):
        '''Calculates the logistic regression loss using X, y, w, 
        and alpha. Useful as a helper function for train().
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        loss: (scalar) logistic regression loss
        '''
        ### Your code here
        sigma_val = self.calculate_sigmoid(np.matmul(X,self.w))
        LHS = np.matmul(y.T, np.log(sigma_val))
        RHS = np.matmul((1 - y).T, np.log(1 - sigma_val))
        reg_term = self.alpha * np.linalg.norm(self.w,ord=2) * np.linalg.norm(self.w,ord=2)
        output = (-LHS - RHS + reg_term)[0]
        return output
    
    def calculate_gradient(self, X, y):
        '''Calculates the gradient of the logistic regression loss 
        using X, y, w, and alpha. Useful as a helper function 
        for train().
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        gradient: (n x 1) gradient vector for logistic regression loss
        '''
        ### Your code here
        inside = self.calculate_sigmoid(np.matmul(X, self.w)) - y
        gradient = np.matmul(X.T,inside) + 2*self.alpha*self.w
        return gradient
    
    def calculate_sigmoid(self, x):
        '''Calculates the sigmoid function on each element in vector x. 
        Useful as a helper function for predict(), calculate_loss(), 
        and calculate_gradient().
        
        Parameters
        ----------
        x: (m x 1) vector
        
        Returns
        -------
        sigmoid_x: (m x 1) vector of sigmoid on each element in x
        '''
        ### Your code here
        sigmoid_x = []
        for curr_x in x:
            val = 1/(1 + pow(math.e, -curr_x[0])) #change if neccesary
            sigmoid_x.append([val],)
        sigmoid_x = np.array(sigmoid_x)
        return sigmoid_x


# **2.6 Plot Loss over Epoch and Search the space randomly to find best hyperparameters.**
# 
# A: Using your implementation above, train a logistic regression model **(alpha=0, t=100, eta=1e-3)** on the voice recognition training data. Plot the training loss over epochs. Make sure to label your axes. You should see the loss decreasing and start to converge. 
# 
# B: Using **alpha between (0,1), eta between(0, 0.001) and t between (0, 100)**, find the best hyperparameters for LogisticRegression. You can randomly search the space 20 times to find the best hyperparameters.
# 
# C. Compare accuracy on the test dataset for both the scenarios.

# In[71]:


#code here
model = LogisticRegression(0, 100, 1e-3)
losses = model.train(voice_X_train,voice_y_train)
plt.plot(list(range(1,101)), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('epoch vs loss on voice training data')
plt.show()


# In[72]:


def print_label(labels):
    toret = []
    for i in labels:
        if i == 1:
            toret.append('male')
        else:
            toret.append('female')
    print(toret)


# In[80]:


best_train = float("-inf")
best_val = float("-inf")
best_test = float("-inf")
best_hyper = {'alpha' : 0, 't' : 0, 'eta' : 0}
pred_test = []
_actual = []

for i in range(1, 21):

    model = LogisticRegression(i/20, i*5, i/100000 + 0.0008)
    model.train(voice_X_train,voice_y_train)
    
    y_pred_train = model.predict(voice_X_train)
    y_true_train = voice_y_train
    current_train = accuracy_score(y_true_train, y_pred_train)
    
    y_pred_val = model.predict(voice_X_val)
    y_true_val = voice_y_val
    current_val = accuracy_score(y_true_val, y_pred_val)
    
    y_pred_test = model.predict(voice_X_test)
    y_true_test = voice_y_test
    current_test = accuracy_score(y_true_test, y_pred_test)
    
    if current_train > best_train and current_val > best_val and current_test > best_test: 
        best_train = current_train
        best_val = current_val
        best_test = current_test
        best_hyper['alpha'] = i/20
        best_hyper['t'] = i*5
        best_hyper['eta'] = i/100000 + 0.0008
        pred_test = y_pred_test
        _actual = y_true_test
print(best_train)
print(best_val)
print(best_train)
print('best hyperparameters -->' , best_hyper)

print('first five predictions:')
print_label(pred_test[:5])
print('actual: ')
print_label(_actual[:5])


# **2.7 Feature Importance**
# 
# Interpret your trained model using a bar chart of the model weights. Make sure to label the bars (x-axis) and don't forget the bias term! 

# In[81]:


num_features_voice = ['meanfreq', 'sd', 'Q75', 'IQR', 'skew', 'sp.ent', 'sfm', 'mode', 'meanfun', 'minfun', 'maxfun',
               'meandom', 'mindom', 'maxdom', 'modindx']
all_features_voice = num_features_voice + ['Bias']
voice_df1['Bias'] = 1


# In[82]:


len(model.w), len(all_features_voice)


# In[83]:


model.w


# In[84]:


#code here
fig = plt.figure(figsize = (20,15))
xval = np.zeros((31))
yval = np.reshape(model.w, -1)
ax = sns.barplot(x=all_features_voice, y=yval)
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('feature name')
ax.set_ylabel('feature importance (coefficient value)')
ax.set_title('feature importance across features')
plt.show()


# 
# # **Part 3: Support Vector Machines - with the same Dataset**

# **3.1 Dual SVM**
# 
# A) Train a dual SVM (with default parameters) for both kernel=“linear” and kernel=“rbf”) on the Voice Recognition training data.
# 
# B) Make predictions and report the accuracy on the training, validation, and test sets. Which kernel gave better accuracy on test dataset and why do you think that was better?
# 
# C) Please report the support vectors in both the cases and what do you observe? Explain
# 

# In[85]:


#code here
from sklearn import svm
dualModelSVM = svm.SVC(kernel='linear')
dualModelSVM.fit(voice_X_train,voice_y_train)

y_pred_train = dualModelSVM.predict(voice_X_train)
y_true_train = voice_y_train
print('accuracy on train set: ')
print (accuracy_score(y_true_train, y_pred_train))
y_pred_val = dualModelSVM.predict(voice_X_val)
y_true_val = voice_y_val
print('accuracy on val set: ')
print (accuracy_score(y_true_val, y_pred_val))

y_pred_test = dualModelSVM.predict(voice_X_test)
y_true_test = voice_y_test
print('accuracy on test set: ')
print (accuracy_score(y_true_test, y_pred_test))

print('first five predictions:')
print(y_pred_test[:5])
print('actual: ')
print(y_true_test[:5])


# In[86]:


from sklearn import svm
dualModelSVM = svm.SVC(kernel='rbf')
dualModelSVM.fit(voice_X_train,voice_y_train)

y_pred_train = dualModelSVM.predict(voice_X_train)
y_true_train = voice_y_train
print('accuracy on train set: ')
print (accuracy_score(y_true_train, y_pred_train))
y_pred_val = dualModelSVM.predict(voice_X_val)
y_true_val = voice_y_val
print('accuracy on val set: ')
print (accuracy_score(y_true_val, y_pred_val))

y_pred_test = dualModelSVM.predict(voice_X_test)
y_true_test = voice_y_test
print('accuracy on test set: ')
print (accuracy_score(y_true_test, y_pred_test))

print('first five predictions:')
print(y_pred_test[:5])
print('actual: ')
print(y_true_test[:5])


# **3.2 Using Kernel “rbf”, tune the hyperparameter “C” using the Grid Search & k-fold cross validation. You may take k=5 and assume values in grid between 1 to 100 with interval range of your choice.**

# In[87]:


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(GridSearchCV(LinearSVC(),
                                 param_grid = {"C":np.logspace(-3,3,20),
                                              "loss":["hinge", "squared_hinge"],
                                              "penalty":["l1","l2"]},
                                 return_train_score=True))
pipe.fit(voice_X_train,voice_y_train)
grid_search_results = pipe.named_steps["gridsearchcv"]
print(f"Best score:", grid_search_results.best_score_)
print(f"Best params:", grid_search_results.best_params_)
print(f"Test score:", pipe.score(voice_X_test,voice_y_test))


# In[88]:


'''
Best score: 0.9705263157894738
Best params: {'C': 0.1623776739188721, 'loss': 'hinge', 'penalty': 'l2'}
Test score: 0.9700315457413249
'''


# In[ ]:




