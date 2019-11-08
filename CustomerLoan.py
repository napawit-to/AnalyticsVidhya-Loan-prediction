#!/usr/bin/env python
# coding: utf-8

# In[279]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle


# In[280]:


import warnings
warnings.filterwarnings("ignore")


# In[281]:


trained_data = "train.csv"
test_data = "test.csv"


# In[282]:


data_raw_trained = pd.read_csv(trained_data,header=0,sep=',',na_values='unknown')
data_raw_trained['Loan_Status'].replace('N', 0, inplace=True)
data_raw_trained['Loan_Status'].replace('Y', 1, inplace=True)


# In[283]:


data_raw_trained.shape


# ## Data preparation

# In[284]:


def preprocessingtrained(data):
    data_raw_trained_clean = data.copy().drop('Loan_ID',axis=1)
    # Seperate between categorical and numerical data
    clean_data_obj = data_raw_trained_clean.select_dtypes(include=['object'])
    clean_data_num = data_raw_trained_clean.select_dtypes(exclude=['object']) 
    # One-hot categorical data
    clean_data_obj = pd.get_dummies(clean_data_obj,drop_first=True)
    scaler = MinMaxScaler()
    scaler.fit(clean_data_num)
    clean_data_num = pd.DataFrame(scaler.transform(clean_data_num),columns=clean_data_num.keys())
    data_raw_clean = clean_data_num.join(clean_data_obj)
    return data_raw_clean


# In[285]:


prep_data_trained = preprocessingtrained(data_raw_trained)
features = prep_data_trained.dropna().drop('Loan_Status',axis=1)
label = prep_data_trained.dropna().Loan_Status


# In[286]:


train_x,test_x,train_y,test_y = train_test_split(features,label, test_size=0.33, random_state=42)


# In[287]:


# Logistic Regression
lr_clf = LogisticRegression()
scores = cross_val_score(lr_clf,train_x,train_y,cv=3,scoring='roc_auc')

print(f"Logistic Regression model's average AUC: {scores.mean():.4f}")


# In[288]:


# Random Forest
rf_clf = RandomForestClassifier()
scores = cross_val_score(rf_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"Random Forest Classfier model's average AUC: {scores.mean():.4f}")


# In[289]:


# Decision Tree
tree_clf = DecisionTreeClassifier()
scores = cross_val_score(tree_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"Tree Classifer model's average AUC: {scores.mean():.4f}")


# In[290]:


# SVM
svm_clf = SVC()
scores = cross_val_score(svm_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"SVM Classfier model's average AUC: {scores.mean():.4f}")


# In[291]:


# XGBClassifier
xg_clf = XGBClassifier()
scores = cross_val_score(xg_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"XG Classfier model's average AUC: {scores.mean():.4f}")


# ## Selected Model Evaluation 

# In[292]:


def eval_matrics(pred_class_y,pred_prob_y,label_y,model_name='Model'):
    # Matrics
    auc = roc_auc_score(label_y,pred_prob_y[:,1])
    print(f'{model_name} AUC is {auc:.4f}')
    acc = accuracy_score(label_y,pred_class_y)
    print(f'{model_name} Accuracy is {acc:.4f}')
    precision = precision_score(label_y,pred_class_y)
    print(f'{model_name} precision is {precision:.4f}')
    recall = recall_score(label_y,pred_class_y)
    print(f'{model_name} recall is {recall:.4f}')
    f1 = f1_score(label_y,pred_class_y)
    print(f'{model_name} f1-score is {f1:.4f}')
    # Confusion Matrix
    print(f'Confusion Matrix of {model_name}:')
    print(confusion_matrix(label_y,pred_class_y))
    return    


# In[293]:


# LogisticRegression
lr_clf = LogisticRegression().fit(train_x,train_y)
pred_class_y = lr_clf.predict(test_x)
pred_prob_y = lr_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'Logistic Regression Model')


# In[294]:


pred_class_y


# In[194]:


fpr_lr_base, tpr_lr_base ,_= roc_curve(test_y, pred_prob_y[:,1])


# In[195]:


# Fit and Predict
rf_clf = RandomForestClassifier().fit(train_x,train_y)
pred_class_y = rf_clf.predict(test_x)
pred_prob_y = rf_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'Random Forest Model')


# In[196]:


# Calculate fpr,and tpr for future ROC ploting
fpr_rf_base, tpr_rf_base ,_= roc_curve(test_y, pred_prob_y[:,1])


# In[197]:



plt.plot(fpr_lr_base, tpr_lr_base ,linewidth=2, label='LR')
plt.plot(fpr_rf_base, tpr_rf_base ,linewidth=2, label='XG')
plt.plot([0,1],[0,1],'k--')
plt.axis([0,1,0,1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Baseline Models')
plt.show()


# ## Gridsearch

# In[198]:


# Create Grid parameters
# linear regularization type (penalty)
penalty_type = ['l1','l2']

# Inverse of regularization strength (C)
reg_str = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 

# Weights associated with classes (class_weight)
cl_weight = [None, 'balanced']
# cl_weight = [None, 'balanced',{0:1,1:2}]

# Create the grid search parameters
lr_param_grid = {'penalty':penalty_type,
                  'C':reg_str,
                  'class_weight':cl_weight}
# Create a based model
lr_clf = LogisticRegression()
# Instantiate the grid search model
lr_grid_search = GridSearchCV(estimator = lr_clf, param_grid = lr_param_grid, cv=3, scoring ='roc_auc',verbose=1)
# Fit the grid search to the data
lr_grid_search.fit(train_x,train_y)


# In[71]:


# Create Grid parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree 
max_depth = [int(x) for x in np.linspace(start = 10, stop =100, num =10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
rf_random_grid = {'n_estimators':n_estimators, 
                  'max_features':max_features,
                  'max_depth':max_depth, 
                  'min_samples_split':min_samples_split,
                  'min_samples_leaf':min_samples_leaf,
                  'bootstrap':bootstrap}


# In[72]:


# Random Search Training
# Create a based model
rf = RandomForestClassifier()
# Initiate a random search model 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rf_random_grid, n_iter = 20, 
                              cv =3, random_state =42,scoring='roc_auc')
# Fit the random search model
rf_random.fit(train_x,train_y)


# ## Evaluate Tuning Models

# In[312]:


lr_grid_search.best_params_


# In[320]:


# Fit and Predict
lr_clf = LogisticRegression(penalty='l2', C=0.01,class_weight='balanced').fit(train_x,train_y)
pred_class_y = lr_clf.predict(test_x)
pred_prob_y = lr_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'Tuned LR Model')


# In[323]:


fpr_lr_tuned, tpr_lr_tuned ,_= roc_curve(test_y, pred_prob_y[:,1])


# In[324]:


rf_random.best_params_


# In[203]:


# Fit and Predict
rf_clf = RandomForestClassifier(n_estimators=2000, bootstrap=False, 
                                  max_depth=20, max_features='auto', 
                                  min_samples_leaf=1, min_samples_split=2, 
                                 ).fit(train_x,train_y)
pred_class_y = rf_clf.predict(test_x)
pred_prob_y = rf_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'Tuned RF Model')


# In[299]:


# Calculate fpr,and tpr for future ROC ploting
fpr_rf_tuned, tpr_rf_tuned ,_= roc_curve(test_y, pred_prob_y[:,1])


# #### Plot ROC of Tuning vs Base Models

# In[300]:


plt.plot(fpr_lr_base, tpr_lr_base, 'b--',linewidth=1, label='LR_based')
plt.plot(fpr_lr_tuned, tpr_lr_tuned, 'b-',linewidth=2, label='LR_tuned')
plt.plot(fpr_rf_base, tpr_rf_base, 'g--',linewidth=1, label='RF_based')
plt.plot(fpr_rf_tuned, tpr_rf_tuned, 'g-',linewidth=2, label='RF_tuned')
plt.plot([0,1],[0,1],'k--')
plt.axis([0,1,0,1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Tuning vs Base Models')
plt.show()


# ## Preprocessing test data

# In[301]:


data_raw_test = pd.read_csv(test_data,header=0,sep=',',na_values='unknown')


# In[302]:


def preprocessingtest(data):
    data_raw_trained_clean = data.copy().drop('Loan_ID',axis=1)
    # Seperate between categorical and numerical data
    clean_data_obj_test = data_raw_trained_clean.select_dtypes(include=['object'])
    clean_data_num_test = data_raw_trained_clean.select_dtypes(exclude=['object']) 
    # One-hot categorical data
    clean_data_obj_test = pd.get_dummies(clean_data_obj_test,drop_first=True)
    scaler = MinMaxScaler()
    scaler.fit(clean_data_num_test)
    clean_data_num_test = pd.DataFrame(scaler.transform(clean_data_num_test),columns=clean_data_num_test.keys())
    data_raw_clean = clean_data_num_test.join(clean_data_obj_test)
    return data_raw_test_clean


# In[303]:


data_raw_test_clean = preprocessingtest(data_raw_test)


# In[304]:


data_raw_test_clean =data_raw_test_clean.fillna(0)
data_raw_test_clean.corr


# In[305]:


features.shape


# In[306]:


prep_data_test_clean.shape


# In[327]:


# Fit and Predict
lr_clf = LogisticRegression(penalty='l2', C=0.01,class_weight='balanced').fit(features,label)
pred_class_y = lr_clf.predict(prep_data_test_clean)


# In[329]:


df = pd.DataFrame(pred_class_y)
df.to_csv(r'status.csv')
# col_names =  ['Loan_Status']
# result_df  = pd.DataFrame(columns = col_names)
# result_df['Loan_Status'] = 


# In[330]:


pred_class_y


# In[ ]:





# In[ ]:




