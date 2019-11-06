#!/usr/bin/env python
# coding: utf-8

# In[270]:


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

import pickle


# In[247]:


import warnings
warnings.filterwarnings("ignore")


# In[248]:


trained_data = "train.csv"
test_data = "test.csv"


# In[249]:


data_raw_trained = pd.read_csv(trained_data,header=0,sep=',',na_values='unknown')
data_raw_trained['Loan_Status'].replace('N', 0, inplace=True)
data_raw_trained['Loan_Status'].replace('Y', 1, inplace=True)

data_raw_test = pd.read_csv(test_data,header=0,sep=',',na_values='unknown')
loanID = data_raw_test.dropna().Loan_ID


# ## Data preparation

# In[250]:


def preprocessing(data):
    data_raw_trained_clean = data.copy().drop('Loan_ID',axis=1)
    # Seperate between categorical and numerical data
    clean_data_obj = data_raw_trained_clean.select_dtypes(include=['object'])
    clean_data_num = data_raw_trained_clean.select_dtypes(exclude=['object']) 
    # One-hot categorical data
    clean_data_obj = pd.get_dummies(clean_data_obj,drop_first=True)
    scaler = MinMaxScaler()
    scaler.fit(clean_data_num)
    clean_data_num = pd.DataFrame(scaler.transform(clean_data_num),columns=clean_data_num.keys())
    data_raw_trained_clean = clean_data_num.join(clean_data_obj)
    return data_raw_trained_clean


# In[251]:


train_x,test_x,train_y,test_y = train_test_split(features,label, test_size=0.33, random_state=42)


# In[252]:


# Logistic Regression
lr_clf = LogisticRegression()
scores = cross_val_score(lr_clf,train_x,train_y,cv=3,scoring='roc_auc')

print(f"Logistic Regression model's average AUC: {scores.mean():.4f}")


# In[253]:


# Decision Tree
tree_clf = DecisionTreeClassifier()
scores = cross_val_score(tree_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"Tree Classifer model's average AUC: {scores.mean():.4f}")


# In[254]:


# SVM
svm_clf = SVC()
scores = cross_val_score(svm_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"SVM Classfier model's average AUC: {scores.mean():.4f}")


# In[255]:


# XGBClassifier
xg_clf = XGBClassifier()
scores = cross_val_score(xg_clf,train_x,train_y,cv=10,scoring='roc_auc')

print(f"XG Classfier model's average AUC: {scores.mean():.4f}")


# ## Selected Model Evaluation 

# In[256]:


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


# In[257]:


# LogisticRegression
lr_clf = LogisticRegression().fit(train_x,train_y)
pred_class_y = lr_clf.predict(test_x)
pred_prob_y = lr_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'Logistic Regression Model')


# In[258]:


fpr_lr_base, tpr_lr_base ,_= roc_curve(test_y, pred_prob_y[:,1])


# In[259]:


# XGBClassifier
xg_clf = XGBClassifier().fit(train_x,train_y)
pred_class_y = xg_clf.predict(test_x)
pred_prob_y = xg_clf.predict_proba(test_x)
# Evaluate
eval_matrics(pred_class_y,pred_prob_y,test_y,'XGBClassifier Model')


# In[260]:


# Calculate fpr,and tpr for future ROC ploting
fpr_xg_base, tpr_xg_base ,_= roc_curve(test_y, pred_prob_y[:,1])


# In[261]:



plt.plot(fpr_lr_base, tpr_lr_base ,linewidth=2, label='LR')
plt.plot(fpr_xg_base, tpr_xg_base ,linewidth=2, label='XG')
plt.plot([0,1],[0,1],'k--')
plt.axis([0,1,0,1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Baseline Models')
plt.show()


# In[216]:


from sklearn.model_selection import train_test_split
data_raw_trained_clean = preprocessing(data_raw_trained)
data_raw_test_clean = preprocessing(data_raw_test)
features = data_raw_trained_clean.dropna().drop('Loan_Status',axis=1)
label = data_raw_trained_clean.dropna().Loan_Status

model = XGBClassifier()
model.fit(features, label)


# In[272]:


y_pred = model.predict(data_raw_test_clean)


# In[218]:





# In[188]:





# In[ ]:




