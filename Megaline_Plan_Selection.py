#!/usr/bin/env python
# coding: utf-8

# ## Megaline Plan Reccomendations (Machine Learning)

# Megaline is a mobile carrier trying to develop a model to analyze subscribers' behavior. The models goal would be to recommend one of Megaline's newer plans: Smart or Ultra. 
# 
# Project success will be in developing a model with the highest possible accuracy (0.75) that will pick the right plan.
# 
# The dataset "user_behavior"(datasets/users_behavior.csv) will be uploaded onto jupyterhub.  Source data will be split into a training set, a validation set, and a test set. An investigation will follow testing the quality of different models (classification: LogisticRegression, DecisionTrees, RandomForest by changing hyperparameters. The findings of each model test will be described. Choice was made not to test any regression models (Linear Regression, Decision Tree Regressor, or Random Forest Regressor) since the models goal was to make a binary choice between two choices (reccomend the Ulta plan or not) rather than a numeric projection. An ultimate winning model will be declared by evaluations using the test set.
# 
# Each row in dataset is for one user.  Features/columns in the dataset are as follows:
# сalls — number of calls,
# minutes — total call duration in minutes,
# messages — number of text messages,
# mb_used — Internet traffic used in MB,
# is_ultra — plan for the current month (Ultra - 1, Smart - 0).

# In[1]:


#import all needed libraries 
import pandas as pd
#import named regression models 
from sklearn.linear_model import LinearRegression

#import ability to split into training and testing data sets 
from sklearn.model_selection import train_test_split

#import ability to evaluate accuracy of data 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#import classification modesl 

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from joblib import dump




# In[2]:


#upload file 
megaline = pd.read_csv('/datasets/users_behavior.csv')


# **PreProcessing 

# In[3]:


megaline.info()


# In[4]:


megaline.head(5)


# In[5]:


megaline.info()


# In[6]:


#check for empty values 
megaline.isna().sum()


# In[7]:


#see any duplicates 
duplicate=megaline[megaline.duplicated()]
duplicate
#seems there are no duplicates 


# In[8]:


#split source data in test, training, and validation set 6:2:2
#create split between 60% assigned to training and 40% assigned to megaline_temp
megaline_train, megaline_temp = train_test_split(megaline, test_size=0.4, random_state=54321) 
#divide megaline_temp between _validation and _test dataframes for the model. This sources 20% of the data for validation and testing. 
megaline_valid, megaline_test = train_test_split(megaline_temp, test_size=0.5, random_state=54321) 


# In[9]:


#test random forest classifier 

#define variables for training 
features_train = megaline_train.drop(['is_ultra'], axis=1)
target_train = megaline_train['is_ultra']
#define variables for testing
features_test = megaline_test.drop(['is_ultra'], axis=1)
target_test = megaline_test['is_ultra']
#define variables for validation 
features_valid = megaline_valid.drop(['is_ultra'], axis=1)
target_valid = megaline_valid['is_ultra']

best_score = 0
best_est = 0
for est in range(1,100): 
    # choose hyperparameter range (tried between 1-1/50/100)
    model = RandomForestClassifier(random_state=54321, n_estimators=est) 
    model.fit(features_train, target_train) 
    score = model.score(features_valid, target_valid) 
    if score > best_score:
        best_score = score
        best_est = est

print("Accuracy of the best model on the validation set (n_estimators = {}): {}".format(best_est, best_score))

final_model = RandomForestClassifier(random_state=54321, n_estimators=52) # change n_estimators to get best model
final_model.fit(features_test, target_test)


# In[18]:


model = RandomForestClassifier(random_state=54321, n_estimators=52)
model.fit(features_train, target_train)
predictions_valid = model.predict(features_valid)
print("Random Forest Validation Accuracy:", accuracy_score(target_valid, predictions_valid))
predictions_test = model.predict(features_test)
print("Random Forest Test Accuracy:", accuracy_score(target_test, predictions_test))


# A random forest classifier was used.  Tested several max parameters between 1 and 100.  The best n_estimator was determines to be 52.  This gave us our best accuracy point at 78.6%.  This is better than random guessing (50/50). When model was run on test dataframe the accuracy increased up to 81.4%.  Wonderful to be above 80%. A positive indication that the model trained well on the train and validation data. 

# In[13]:


#test Decision Tree Classifier 

#define variables for training 
features_train = megaline_train.drop(['is_ultra'], axis=1)
target_train = megaline_train['is_ultra']
#define variables for testing
features_test = megaline_test.drop(['is_ultra'], axis=1)
target_test = megaline_test['is_ultra']
#define variables for validation 
features_valid = megaline_valid.drop(['is_ultra'], axis=1)
target_valid = megaline_valid['is_ultra']

best_model = None
best_result = 0
for depth in range(1, 100):
	model = DecisionTreeClassifier(random_state=12345, max_depth=depth) # create a model with the given depth
	model.fit(features_train, target_train)
	predictions = model.predict(features_valid)
	result = accuracy_score(target_valid,predictions)
	if result > best_result:
		best_model = model
		best_result = result
        
print("Accuracy of the best model:", best_result)
print("Best depth:", depth)


# In[14]:


decision_model = DecisionTreeClassifier(random_state=54321, max_depth=99)
decision_model.fit(features_train, target_train)
decision_predictions_valid = model.predict(features_valid)
print("Decision Tree Validation Accuracy:", accuracy_score(target_valid, predictions_valid))
predictions_test = model.predict(features_test)
print("Decision Tree Test Accuracy:", accuracy_score(target_test, predictions_test))


# Utilizing the decision tree classifier the most accurate achieved was 77.9%.  This is above random 50/50% but in no way a good model.  I tested the depth at different parameters (6,10,50,100). The accuracy never varied.  The best depth was consistently one number before the tested parameters. When the model was ran on the test data the accuracy was worse than it performed on the validation data.  This has potential to be a case of underfitting.  

# In[16]:


#test Logistic  regression model 

#define variables for training 
features_train = megaline_train.drop(['is_ultra'], axis=1)
target_train = megaline_train['is_ultra']
#define variables for testing
features_test = megaline_test.drop(['is_ultra'], axis=1)
target_test = megaline_test['is_ultra']
#define variables for validation 
features_valid = megaline_valid.drop(['is_ultra'], axis=1)
target_valid = megaline_valid['is_ultra']

#train and validate model 
logistic_model =LogisticRegression(random_state=54321, solver='liblinear') 
logistic_model.fit(features_train, target_train)
score_train = logistic_model.score(features_train, target_train)  
score_valid = logistic_model.score(features_valid, target_valid)  

print(
    "Accuracy of the logistic regression model on the training set:",
    score_train,
)
print(
    "Accuracy of the logistic regression model on the validation set:",
    score_valid,
)


# In[19]:


#test logistic regression model on test set
logistic_predictions_test = logistic_model.predict(features_test)
print("Accuracy of Logistic Regression on Test Set:", accuracy_score(target_test, logistic_predictions_test))


# Linear Regression performed the worst out of the three models tested.  On validation set it only achieved 67.8% accuracy which is only slight improvement from guessing. On the test set the accuracy increased to 74.8%.  However this is the result that decision tree and random forest classifiers achieved on their training/validation sets.  Performance on the test set was even higher. 

# General Conclusion:
#     Megaline has several options on models to use to predict customer reccomendations.  Evaluated in this project were Decision Trees, Random Forest, and Logistic Regression.  Each model was trained, validated, and tested with their own slice of the data from source file.  The accuracies from the test set were as follows: Logistic Regression= 74%, Decision Tree=74.8%, and Random Forest=81.5%.  The superior model Megaline should use to evaluate all future customers is the random forest since the model achieved the highest accuracy in testing. This achieved the set standard of above 75%.  Neither Logistic regression or decision tree was able to perform at this higher level.  Machine Learning made a savy business reccomendation.  
