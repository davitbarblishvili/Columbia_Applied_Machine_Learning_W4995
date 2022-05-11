#!/usr/bin/env python
# coding: utf-8

# # Homework 3
# 
# ## Part 1: Imbalanced Dataset
# This part of homework helps you practice to classify a highly imbalanced dataset in which the number of examples in one class greatly outnumbers the examples in another. You will work with the Credit Card Fraud Detection dataset hosted on Kaggle. The aim is to detect a mere 492 fraudulent transactions from 284,807 transactions in total. 
# 
# ### Instructions
# 
# Please push the .ipynb, .py, and .pdf to Github Classroom prior to the deadline. Please include your UNI as well.
# 
# Due Date : TBD
# 
# ### Davit Barblishvili
# 
# ### DB3230
# 
# ## 0 Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


# ## 1 Data processing and exploration
# Download the Kaggle Credit Card Fraud data set. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# In[2]:


raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()


# ### 1.1 Examine the class label imbalance
# Let's look at the dataset imbalance:
# 
# **Q1. How many observations are there in this dataset? How many of them have positive label (labeled as 1)?**

# In[3]:


# Your Code Here
import collections
target = raw_df.values[:,-1]
counter = collections.Counter(target)
for k,v in counter.items():
    per = v / len(target) * 100
    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))


# In[4]:


print(f'Classes Count in Credit card Fraud Dataset \n', pd.value_counts(raw_df['Class'], sort = True).sort_index())
credit_classes = pd.value_counts(raw_df['Class'], sort = True).sort_index()
credit_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# # Answer
# 
# ***From the previous result, we can see that there are two observations, or in other words two different classes. 
# First class is '0' which is a majority class and second one is class '1' which is minority class. We have very
# skewed dataset. Only 0.173% of total data belongs to class '1' which is not a data distribution we should have
# before doing any modeling.***

# ### 1.2 Clean, split and normalize the data
# The raw data has a few issues. First the `Time` and `Amount` columns are too variable to use directly. Drop the `Time` column (since it's not clear what it means) and take the log of the `Amount` column to reduce its range.

# In[5]:


cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)


# In[6]:


cleaned_df


# **Q2. Split the dataset into development and test sets. Please set test size as 0.2 and random state as 42.**

# In[7]:


# Your Code Here
X=cleaned_df.drop(['Class'],axis=1)
y=cleaned_df['Class']

X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# **Q3. Normalize the input features using the sklearn StandardScaler. Print the shape of your development features and test features.**

# In[8]:


# Your Code Here
scaler = StandardScaler()
X_dev = scaler.fit_transform(X_dev)
X_test = scaler.transform(X_test)


# In[9]:


print(X_dev.shape)
print(X_test.shape)


# ### 1.3 Define the model and metrics
# **Q4. First, fit a default logistic regression model. Print the AUC and average precision of 5-fold cross validation.**

# In[10]:


# Your Code Here
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()
logreg.fit(X_dev,y_dev)

y_pred_default=logreg.predict_proba(X_test)[::,1]
y_pred_default_cfmatrix=logreg.predict(X_test)
y_pred_default


# In[11]:


from sklearn import metrics


#y_pred_default = logreg.predict_proba(X_test)[::,1]
auc_default_legreg = metrics.roc_auc_score(y_test, y_pred_default)
fpr_default, tpr_default, _ = metrics.roc_curve(y_test,  y_pred_default)
print(auc_default_legreg)


# In[12]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[13]:


kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold, scoring="average_precision")
cv_leg_reg_default_mean = results.mean()*100.0
cv_leg_reg_default_std = results.std()*100.0

# Output the accuracy. Calculate the mean and std across all folds. 
print("Accuracy: %.3f%% (%.3f%%)" % (cv_leg_reg_default_mean, cv_leg_reg_default_std))


# **Q5.1. Perform random under sampling on the development set. What is the shape of your development features? How many  positive and negative labels are there in your development set? (Please set random state as 42 when performing random under sampling)**

# In[14]:


# Your Code Here

rus = RandomUnderSampler(random_state=42) 
# resampling X, y
X_rus, y_rus = rus.fit_resample(X_dev, y_dev)
# new class distribution
print(collections.Counter(y_rus))


# In[15]:


print(X_rus.shape)
print(y_rus.shape)


# # Answer
# 
# ***From the above operation, we can see that we have 50/50 distribution of positive (1) and negative (0) classes which I think is much better than what we originally had.*** 

# **Q5.2. Fit a default logistic regression model using under sampling. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing random under sampling)**

# In[16]:


logreg=LogisticRegression()
logreg.fit(X_rus,y_rus)
y_pred_rus=logreg.predict_proba(X_test)[::,1]
y_pred_rus_cfmatrix=logreg.predict(X_test)


# In[17]:


auc_rus = metrics.roc_auc_score(y_test, y_pred_rus)
fpr_rus, tpr_rus, _ = metrics.roc_curve(y_test,  y_pred_rus)
print(auc_rus)


# In[18]:


kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results_rus = cross_val_score(model, X_rus, y_rus, cv=kfold, scoring="average_precision")
cv_leg_reg_rus_mean = results_rus.mean()*100.0
cv_leg_reg_rus_std = results_rus.std()*100.0

# Output the accuracy. Calculate the mean and std across all folds. 
print("Accuracy: %.3f%% (%.3f%%)" % (cv_leg_reg_rus_mean, cv_leg_reg_rus_std))


# **Q6.1. Perform random over sampling on the development set. What is the shape of your development features? How many positive and negative labels are there in your development set? (Please set random state as 42 when performing random over sampling)**

# In[19]:


# Your Code Here
ros = RandomOverSampler(random_state=42) 
# resampling X, y
X_ros, y_ros = ros.fit_resample(X_dev, y_dev)
# new class distribution
print(collections.Counter(y_ros))


# In[20]:


print(X_ros.shape)
print(y_ros.shape)


# # Answer
# ***definitely the number of samples we have is much larger than we had during undersampling and it is logical. However, the distribution of 1 and 0 classes is still 50/50 which is still better than original imbalanced dataset***

# **Q6.2. Fit a default logistic regression model using over sampling. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing random over sampling)**

# In[21]:


# Your Code Here
logreg=LogisticRegression()
logreg.fit(X_ros,y_ros)
y_pred_ros=logreg.predict_proba(X_test)[::,1]
y_pred_ros_cfmatrix=logreg.predict(X_test)


# In[22]:


auc_ros = metrics.roc_auc_score(y_test, y_pred_ros)
fpr_ros, tpr_ros, _ = metrics.roc_curve(y_test,  y_pred_ros)
print(auc_ros)


# In[23]:


kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results_ros = cross_val_score(model, X_ros, y_ros, cv=kfold, scoring="average_precision")
cv_leg_reg_ros_mean = results_ros.mean()*100.0
cv_leg_reg_ros_std = results_ros.std()*100.0

# Output the accuracy. Calculate the mean and std across all folds. 
print("Accuracy: %.3f%% (%.3f%%)" % (cv_leg_reg_ros_mean, cv_leg_reg_ros_std))


# **Q7.1. Perform Synthetic Minority Oversampling Technique (SMOTE) on the development set. What is the shape of your development features? How many positive and negative labels are there in your development set? (Please set random state as 42 when performing SMOTE)**

# In[24]:


# Your Code Here
oversample = SMOTE(random_state=42)
X_smote, y_smote = oversample.fit_resample(X_dev, y_dev)


# In[25]:


print(X_smote.shape)
print(y_smote.shape)


# In[26]:


print(collections.Counter(y_smote))


# # Answer
# ***we have the exact same number of negative and positive values however the number of samples is much larger***

# **Q7.2. Fit a default logistic regression model using SMOTE. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing SMOTE)**

# In[27]:


# Your Code Here
logreg = LogisticRegression()
logreg.fit(X_smote,y_smote)
y_pred_smote = logreg.predict_proba(X_test)[::,1]
y_pred_smote_cfmatrix = logreg.predict(X_test)
y_pred_smote


# In[28]:


auc_smote = metrics.roc_auc_score(y_test, y_pred_smote)
fpr_smote, tpr_smote, _ = metrics.roc_curve(y_test,  y_pred_smote)
print(auc_smote)


# In[29]:


kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results_smote = cross_val_score(model, X_smote, y_smote, cv=kfold, scoring="average_precision")
cv_leg_reg_smote_mean = results_smote.mean()*100.0
cv_leg_reg_smote_std = results_smote.std()*100.0

# Output the accuracy. Calculate the mean and std across all folds. 
print("Accuracy: %.3f%% (%.3f%%)" % (cv_leg_reg_smote_mean, cv_leg_reg_smote_std))


# **Q8. Plot confusion matrices on the test set for all four models above. Comment on your result.**

# In[30]:


# Your Code Here
# matrix 1
from sklearn.metrics import confusion_matrix
import seaborn as sns


cf_matrix_1 = confusion_matrix(y_test, y_pred_default_cfmatrix)
ax = sns.heatmap(cf_matrix_1/np.sum(cf_matrix_1),fmt='.2%',  annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[31]:


# Your Code Here
# matrix 2
from sklearn.metrics import confusion_matrix
import seaborn as sns


cf_matrix_2 = confusion_matrix(y_test, y_pred_rus_cfmatrix)
ax = sns.heatmap(cf_matrix_2/np.sum(cf_matrix_2),fmt='.2%', annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[32]:


# Your Code Here
# matrix 3
from sklearn.metrics import confusion_matrix
import seaborn as sns


cf_matrix_3 = confusion_matrix(y_test, y_pred_ros_cfmatrix)
ax = sns.heatmap(cf_matrix_3/np.sum(cf_matrix_3),fmt='.2%', annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[33]:


# Your Code Here
# matrix 4
from sklearn.metrics import confusion_matrix
import seaborn as sns


cf_matrix_4 = confusion_matrix(y_test, y_pred_smote_cfmatrix)
ax = sns.heatmap(cf_matrix_4/np.sum(cf_matrix_4),fmt='.2%', annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# **Q9. Plot the ROC for all four models above in a single plot. Make sure to label the axes and legend. Comment on your result.**

# In[34]:


#create ROC curve
plt.figure(figsize=(15,15),)
plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



plt.plot(fpr_default,tpr_default,label="AUC_default="+str(auc_default_legreg), color='red', linewidth=2)
plt.plot(fpr_rus,tpr_rus,label="AUC_rus="+str(auc_rus), color='blue', linewidth=2)
plt.plot(fpr_ros,tpr_ros,label="AUC_ros="+str(auc_ros), color='green', linewidth=2)
plt.plot(fpr_smote,tpr_smote,label="AUC_smote="+str(auc_smote), color='black', linewidth=2)
plt.legend(loc=4)




plt.show()


# **Q10. Plot the precision-recall curve for all four models above in a single plot. Make sure to label the axes and legend. Comment on your result.**

# In[35]:


# Your Code Here
#calculate precision and recall
from sklearn.metrics import precision_recall_curve
from matplotlib.pyplot import figure

precision_default, recall_default,_ = precision_recall_curve(y_test, y_pred_default)
precision_rus, recall_rus,_ = precision_recall_curve(y_test, y_pred_rus)
precision_ros, recall_ros,_ = precision_recall_curve(y_test, y_pred_ros)
precision_smote, recall_smote,_ = precision_recall_curve(y_test, y_pred_smote)

#create precision recall curve
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(recall_default, precision_default, color='purple')
ax.plot(recall_rus, precision_rus, color='green')
ax.plot(recall_ros, precision_ros, color='blue')
ax.plot(recall_smote, precision_smote, color='red')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()


# **Q11. Adding class weights to a logistic regression model. Print the AUC and average precision of 5-fold cross validation. Also, plot its confusion matrix on test set.**

# In[36]:


# Your Code Here

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(solver='newton-cg')
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr, 
                          param_grid= param_grid,
                          cv=5, 
                          n_jobs=-1, 
                          scoring='roc_auc', 
                          verbose=2).fit(X_dev, y_dev)


# In[37]:


#Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)


# In[38]:


#auc_smote = metrics.roc_auc_score(y_test, y_pred_smote)
gridsearch.fit(X_dev,y_dev)
y_pred_balanced=logreg.predict_proba(X_test)[::,1]
y_pred_balanced_cfmatrix=logreg.predict(X_test)


# In[39]:


auc_balanced = metrics.roc_auc_score(y_test, y_pred_balanced)
fpr_balanced, tpr_balanced, _ = metrics.roc_curve(y_test,  y_pred_balanced)
print(auc_balanced)


# In[40]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix_balanced = confusion_matrix(y_test, y_pred_balanced_cfmatrix)
ax = sns.heatmap(cf_matrix_balanced/np.sum(cf_matrix_balanced),fmt='.2%', annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# **Q12. Plot the ROC and the precision-recall curve for default Logistic without any sampling method and this balanced Logistic model in two single plots. Make sure to label the axes and legend. Comment on your result.**

# In[41]:


# Your Code Here
# default logistic
#create ROC curve
plt.figure(figsize=(15,15),)
plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.plot(fpr_default,tpr_default,label="AUC_default="+str(auc_default_legreg), color='red', linewidth=2)
plt.plot(fpr_balanced,tpr_balanced,label="AUC_balanced="+str(auc_balanced), color='blue', linewidth=2)

plt.legend(loc=4)




plt.show()


# In[42]:


# Your Code Here
#calculate precision and recall
from sklearn.metrics import precision_recall_curve
from matplotlib.pyplot import figure

precision_default, recall_default,_ = precision_recall_curve(y_test, y_pred_default)
precision_balanced, recall_balanced,_ = precision_recall_curve(y_test, y_pred_balanced)


#create precision recall curve
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(recall_default, precision_default, color='purple')
ax.plot(recall_balanced, precision_balanced, color='green')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()


# similar results in the end


# ## Part 2: Unsupervised Learning
# 
# In this part, we will be applying unsupervised learning approaches to a problem in computational biology. Specifically, we will be analyzing single-cell genomic sequencing data. Single-cell genomics is a set of revolutionary new technologies which can profile the genome of a specimen (tissue, blood, etc.) at the resolution of individual cells. This increased granularity can help capture intercellular heterogeneity, key to better understanding and treating complex genetic diseases such as cancer and Alzheimer's. 
# 
# <img src="https://cdn.10xgenomics.com/image/upload/v1574196658/blog/singlecell-v.-bulk-image.png" width="800px"/>
# 
# <center>Source: 10xgenomics.com/blog/single-cell-rna-seq-an-introductory-overview-and-tools-for-getting-started</center>
# 
# A common challenge of genomic datasets is their high-dimensionality: a single observation (a cell, in the case of single-cell data) may have tens of thousands of gene expression features. Fortunately, biology offers a lot of structure - different genes work together in pathways and are co-regulated by gene regulatory networks. Unsupervised learning is widely used to discover this intrinsic structure and prepare the data for further analysis.

# ### Dataset: single-cell RNASeq of mouse brain cells

# We will be working with a single-cell RNASeq dataset of mouse brain cells. In the following gene expression matrix, each row represents a cell and each column represents a gene. Each entry in the matrix is a normalized gene expression count - a higher value means that the gene is expressed more in that cell. The dataset has been pre-processed using various quality control and normalization methods for single-cell data. 
# 
# Data source is on Coursework.

# In[43]:


cell_gene_counts_df = pd.read_csv('data/mouse_brain_cells_gene_counts.csv', index_col='cell')
cell_gene_counts_df


# Note the dimensionality - we have 1000 cells (observations) and 18,585 genes (features)!
# 
# We are also provided a metadata file with annotations for each cell (e.g. cell type, subtissue, mouse sex, etc.)

# In[44]:


cell_metadata_df = pd.read_csv('data/mouse_brain_cells_metadata.csv')
cell_metadata_df


# Different cell types

# In[45]:


cell_metadata_df['cell_ontology_class'].value_counts()


# Different subtissue types (parts of the brain)

# In[46]:


cell_metadata_df['subtissue'].value_counts()


# Our goal in this exercise is to use dimensionality reduction and clustering to visualize and better understand the high-dimensional gene expression matrix. We will use the following pipeline, which is common in single-cell analysis:
# 1. Use PCA to project the gene expression matrix to a lower-dimensional linear subspace.
# 2. Cluster the data using K-means on the first 20 principal components.
# 3. Use t-SNE to project the first 20 principal components onto two dimensions. Visualize the points and color by their clusters from (2).

# ## 1 PCA

# **Q1. Perform PCA and project the gene expression matrix onto its first 50 principal components. You may use `sklearn.decomposition.PCA`.**

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[48]:


### Your code here
pca = PCA(n_components=50)
principalComponents = pca.fit_transform(cell_gene_counts_df)
principalDf = pd.DataFrame(data = principalComponents)
principalDf.head()


# **Q2. Plot the cumulative proportion of variance explained as a function of the number of principal components. How much of the total variance in the dataset is explained by the first 20 principal components?**

# In[49]:


### Your code here

fig = plt.figure(figsize=(12,8))
top_20_pca_var = pca.explained_variance_ratio_[:20]
ax = fig.add_subplot(1,1,1)
plt.plot(np.arange(1,21), top_20_pca_var.cumsum()*100)
ax.set_xlabel("# of PCs")
ax.set_ylabel("% of variance explained")


# **Q3. For the first principal component, report the top 10 loadings (weights) and their corresponding gene names.** In other words, which 10 genes are weighted the most in the first principal component?

# In[50]:


### Your code here
weights = pca.components_
first_component = abs(weights[0])
ind = np.argpartition(first_component, -10)[-10:]

col_names = cell_gene_counts_df.columns[ind]
col_names
print(col_names)


# **Q4. Plot the projection of the data onto the first two principal components using a scatter plot.**

# In[51]:


### Your code here
plt.scatter(x=principalDf[0], y=principalDf[1])
plt.xlabel("PCA #1")
plt.ylabel("PCA #2")
plt.title('Two PCAs for gene expression data')


# **Q5. Now, use a small multiple of four scatter plots to make the same plot as above, but colored by four annotations in the metadata: cell_ontology_class, subtissue, mouse.sex, mouse.id. Include a legend for the labels.** For example, one of the plots should have points projected onto PC 1 and PC 2, colored by their cell_ontology_class.

# In[52]:


### Your code here
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20,45))
cell_ontology_colors = ['red', 'blue', 'yellow', 'purple', 'black', 'green', 'orange']
cell_ontology_colors_map = {
  "oligodendrocyte": "red",
  "endothelial cell": "blue",
  "astrocyte": "purple",
  "neuron": "black",
  "brain pericyte": "green",
  "oligodendrocyte precursor cell": "orange",
  "Bergmann glial cell": "yellow",
}

mouse_sex_colors_map = {
  "F": "red",
  "M": "blue"
}

mouse_id_colors_map = {
  "3_10_M": "red",
  "3_9_M": "blue",
  "3_38_F": "purple",
  "3_8_M": "black",
  "3_11_M": "green",
  "3_39_F": "orange",
  "3_56_F": "yellow",
}

cell_subtissue_colors_map = {
  "Cortex": "red",
  "Hippocampus": "blue",
  "Striatum": "purple",
  "Cerebellum": "black",
}

cell_metadata_df['ontology_color'] = cell_metadata_df.apply(lambda row : cell_ontology_colors_map[row['cell_ontology_class']],axis=1)
cell_metadata_df['sex_color'] = cell_metadata_df.apply(lambda row : mouse_sex_colors_map[row['mouse.sex']],axis=1)
cell_metadata_df['id_color'] = cell_metadata_df.apply(lambda row : mouse_id_colors_map[row['mouse.id']],axis=1)
cell_metadata_df['subtissue_color'] = cell_metadata_df.apply(lambda row : cell_subtissue_colors_map[row['subtissue']],axis=1)

ax1.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['ontology_color'])
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_title("Cell ontology type and first two PCs")
ax2.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['sex_color'])
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_title("Mouse.sex and first two PCs")
ax3.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['id_color'])
ax3.set_xlabel("PC 1")
ax3.set_ylabel("PC 2")
ax3.set_title("Mouse.id and first two PCs")
ax4.scatter(x=principalDf[0], y=principalDf[1], c=cell_metadata_df['subtissue_color'])
ax4.set_xlabel("PC 1")
ax4.set_ylabel("PC 2")
ax3.set_title("subtissue and first two PCs")


fig.tight_layout()


# **Q6. Based on the plots above, the first two principal components correspond to which aspect of the cells? What is the intrinsic dimension that they are describing?**

# ### Your answer here
# PC1 and PC2 are able to distinguish well between cell ontology type 
# and M vs Female best, so I think that these two components correpond to 
# these aspects of the cells

# ## Part 2: K-means

# While the annotations provide high-level information on cell type (e.g. cell_ontology_class has 7 categories), we may also be interested in finding more granular subtypes of cells. To achieve this, we will use K-means clustering to find a large number of clusters in the gene expression dataset. Note that the original gene expression matrix had over 18,000 noisy features, which is not ideal for clustering. So, we will perform K-means clustering on the first 20 principal components of the dataset.

# **Q7. Implement a `kmeans` function which takes in a dataset `X` and a number of clusters `k`, and returns the cluster assignment for each point in `X`. You may NOT use sklearn for this implementation. Use lecture 6, slide 14 as a reference.**

# In[53]:


import random
from scipy.spatial import distance


# In[54]:




def kmeans(X, k, iters=10):
    '''Groups the points in X into k clusters using the K-means algorithm.

    Parameters
    ----------
    X : (m x n) data matrix
    k: number of clusters
    iters: number of iterations to run k-means loop

    Returns
    -------
    y: (m x 1) cluster assignment for each point in X
    '''
    ### Your code here
    count = 0  
    m = len(X)
    idx = np.random.choice(m, k, replace=False)
    n = len(X[0])
    centroids = X[idx, :]
    distances = distance.cdist(X, centroids, 'euclidean')
    min_ks = np.array([np.argmin(i) for i in distances])
    
    while count < iters: 
        centroids = []
        for idx in range(k):
            centroids.append(X[min_ks==idx].mean(axis=0))

        centroids = np.vstack(centroids)
        distances = distance.cdist(X, centroids ,'euclidean')
        min_ks = np.array([np.argmin(i) for i in distances])
        
        count = count +1
        
    return min_ks


# Before applying K-means on the gene expression data, we will test it on the following synthetic dataset to make sure that the implementation is working.

# In[55]:


np.random.seed(0)
x_1 = np.random.multivariate_normal(mean=[1, 2], cov=np.array([[0.8, 0.6], [0.6, 0.8]]), size=100)
x_2 = np.random.multivariate_normal(mean=[-2, -2], cov=np.array([[0.8, -0.4], [-0.4, 0.8]]), size=100)
x_3 = np.random.multivariate_normal(mean=[2, -2], cov=np.array([[0.4, 0], [0, 0.4]]), size=100)
X = np.vstack([x_1, x_2, x_3])

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)


# **Q8. Apply K-means with k=3 to the synthetic dataset above. Plot the points colored by their K-means cluster assignments to verify that your implementation is working.**

# In[56]:


### Your code here
label = kmeans(X, 3, 25)

for i in np.unique(label):
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
plt.legend()
plt.title('synthetic data kmeans = 3')
plt.show()


# **Q9. Use K-means with k=20 to cluster the first 20 principal components of the gene expression data.**

# In[57]:


### Your code here
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(cell_gene_counts_df)
principalDf_20 = pd.DataFrame(data = principalComponents)

principalDf_20.head()


# In[58]:


principalDf_20_numpy = principalDf_20.to_numpy()

pca_labels = kmeans(principalDf_20_numpy, 20, 15)


# In[59]:


for i in np.unique(pca_labels):
    plt.scatter(principalDf_20_numpy[pca_labels == i , 0] , principalDf_20_numpy[pca_labels == i , 1] , label = i)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('kmeans on top 20 on top 20 pca components of gene expression data')
plt.legend()
plt.show()


# ## 3 t-SNE

# In this final section, we will visualize the data again using t-SNE - a non-linear dimensionality reduction algorithm. You can learn more about t-SNE in this interactive tutorial: https://distill.pub/2016/misread-tsne/.

# **Q10. Use t-SNE to reduce the first 20 principal components of the gene expression dataset to two dimensions. You may use `sklearn.manifold.TSNE`.** Note that it is recommended to first perform PCA before applying t-SNE to suppress noise and speed up computation.

# In[60]:


### Your code here
tsne = TSNE()
tsne_pca_results = tsne.fit_transform(principalDf_20)


# **Q11. Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions.**

# In[61]:


### Your code here
plt.figure(figsize=(8, 5))
plt.scatter(tsne_pca_results[:, 0], tsne_pca_results[:, 1], s=10)
plt.xlabel('$TSNE_1$', fontsize=15)
plt.ylabel('$TSNE_2$', fontsize=15)
plt.title('TSNE1 and TSNE2 for PCA20 dimensions from gene expression data')


# **Q12. Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions, with points colored by their cluster assignments from part 2.**

# In[62]:


### Your code herefor i in np.unique(pca_labels):
from matplotlib.pyplot import figure

for i in np.unique(pca_labels):
    plt.scatter(tsne_pca_results[pca_labels == i , 0] , tsne_pca_results[pca_labels == i , 1] , label = i)
plt.xlabel('$TSNE_1$', fontsize=15)
plt.ylabel('$TSNE_2$', fontsize=15)
plt.title('kmeans on top 20 on top 20 pca components of gene expression data then tSNE')
plt.legend()
plt.show()


# **Q13. Why is there overlap between points in different clusters in the t-SNE plot above?**

# ### Your answer here
# There is overlap because we are reducing an already reduced dimensionality of PCA further, so we 
# are unable to see the 'depth' or third/more dimensions which may be separating the dataset. Also, 
# tSNE is a probabilisitic algorithm so it's possible that the overlap is due to the probability based nature -
# which lends itself to non-clear cut /black and white slices.

# These 20 clusters may correspond to various cell subtypes or cell states. They can be further investigated and mapped to known cell types based on their gene expressions (e.g. using the K-means cluster centers). The clusters may also be used in downstream analysis. For instance, we can monitor how the clusters evolve and interact with each other over time in response to a treatment.
