#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Breast cancer prediction


# In[3]:


#Scaling+ PCA+ Classifiers+ Hyperparameter tuning
 
#There are big differences in the accuracy score between different scaling methods for a given classifier 
#we used MinMaxScaler.


# In[4]:


#import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve
#warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[5]:


#Loading the cancer dataset in csv format 
df = pd.read_csv('data.csv')
df.head(7)


# In[6]:


df.shape


# In[7]:


#For finding the empty values in each coloumns
df.isna().sum()


# In[8]:


#Drop the column with all missing values
df = df.dropna(axis=1)


# In[9]:


print(df)


# In[10]:


df.shape


# In[11]:


#number of 'M' & 'B' cell coumt
df['diagnosis'].value_counts()


# In[12]:


#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")


# In[13]:


#Look at the data types 
df.dtypes


# In[14]:


#Encoding categorical values (For transforming M='1' and B='0') 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))


# In[15]:


#pairplot function will create a grid of Axes such that each numeric variable in data and
#It will by shared across the y-axes across a single row and the x-axes across a single column.


# In[16]:


#creating pairplot
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

g = sns.PairGrid(df.iloc[:, 0:11], hue="diagnosis", palette="Set2")
g = g.map_diag(plt.hist, edgecolor="b")
g = g.map_offdiag(plt.scatter, edgecolor="b", s=40)
plt.show()


# In[18]:


#Get the correlation of the columns
df.iloc[:, 1:12].corr()


# In[19]:


#correlation-visualization  
plt.figure(figsize=(50,50))
sns.heatmap(df.iloc[:, 1:].corr(), annot=True, fmt= '.0%',cmap='coolwarm')


# In[20]:


#Drop the column with all missing values
df.drop('id',axis=1,inplace=True)
#df.drop('Unnamed: 32',axis=1,inplace=True)


# In[21]:


#splitting the data set into X and Y
X=df.drop(['diagnosis'],axis=1).values
Y = df['diagnosis'].values


# In[22]:


print(X)


# In[23]:


X.shape


# In[24]:


# In this case I used MinMaxScaler for preprocesing, PCA for feature selection 
# And LogisticRegression,DecisionTree,RandomForest,KNeighbors,SVM_linear,SVM_rbf,GaussianNB classifiers for classification.


# In[25]:


#Split the data again, but this time into 75% training and 25% testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)


# In[26]:


#Feature scaling - Using MinMax scalar, MinMaxScaler rescales the data set such that all feature values are in the range [0, 1]


# In[27]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train.max(axis=0), X_train.min(axis=0))
print(X_train_std.max(axis=0), X_train_std.min(axis=0))
print(X_test.max(axis=0), X_test.min(axis=0))
print(X_test_std.max(axis=0), X_test_std.min(axis=0))


# In[28]:


# Applying Principal component analysis (PCA),
#The main goal of a PCA analysis is to identify patterns in data. PCA aims to detect the correlation between variables. 


# In[29]:


#Applying PCA
from sklearn.decomposition import PCA 
pca = PCA(
    n_components=(11),
    random_state=0
)

pca.fit(X)
x_pca = pca.transform(X)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# In[30]:


x_pca.shape


# In[31]:


print(Y)


# In[32]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


ax = plt.figure(figsize=(8,5))
sns.scatterplot(x_pca[:,0], x_pca[:,1], hue=df['diagnosis'],palette ='binary' )
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[35]:


ax = plt.figure(figsize=(8,5))
sns.scatterplot(x_pca[:,2], x_pca[:,3], hue=df['diagnosis'],palette ='binary' )
plt.xlabel('Third Principal Component')
plt.ylabel('Fourth Principal Component')


# In[36]:


print(pca.explained_variance_ratio_)
print('\nPCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))


# In[37]:


# acc list stores the output of each model
acc = []


# #  (1) LogisticRegression

# In[38]:


from sklearn.linear_model import LogisticRegression
log= LogisticRegression(
C=100,
    penalty='l1',
    solver='liblinear',
    multi_class='ovr',
    random_state=42
)
model1=log.fit(X_train_pca, Y_train)
prediction_test1 = log.predict(X_test_pca)
prediction_train1=log.predict(X_train_pca)
log_reg_predict_proba = log.predict_proba(X_test_pca)[:, 1]


# In[39]:


log.get_params().keys()


# In[40]:


ac1 = accuracy_score(Y_test,prediction_test1)
acc.append(ac1)


# In[41]:


print(classification_report(Y_train, prediction_train1))
print()
print('accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train1) * 100))


# In[42]:


print('LogisticRegression Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test1) * 100))
#print('LogisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, log_reg_predict_proba) * 100))
print('LogisticRegression Test Classification report:\n\n', classification_report(Y_test, prediction_test1))
print('LogisticRegression Training set score: {:.2f}%'.format(log.score(X_train_pca, Y_train) * 100))
print('LogisticRegression Testing set score: {:.2f}%'.format(log.score(X_test_pca, Y_test) * 100))


# In[43]:


# Receiver Operating Characteristic (ROC)
# ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.


# In[44]:


fpr, tpr, thresholds = roc_curve(Y_test, log_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for LogisticRegression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[45]:


# The confusion matrix and the accuracy of the models on the test data
cm=confusion_matrix(Y_test,prediction_test1)
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print()
print('Model LogisticRegression Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[46]:


sns.heatmap(cm,annot=True)
print('LogisticRegression:')
plt.savefig('h.png')


# #  (2) DecisionTreeClassifier

# In[47]:


from sklearn.tree import DecisionTreeClassifier

#dtc=DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtc=DecisionTreeClassifier(criterion = 'entropy',max_depth = 10,random_state =  42)
model2=dtc.fit(X_train_pca,Y_train)
prediction_train2=dtc.predict(X_train_pca)
prediction_test2=dtc.predict(X_test_pca)
dtc_reg_predict_proba = dtc.predict_proba(X_test_pca)[:, 1]


# In[48]:


ac2 = accuracy_score(Y_test,prediction_test2)
acc.append(ac2)


# In[49]:


print(classification_report(Y_train, prediction_train2))
print()
print('DecisionTreeClassifier Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train2) * 100))


# In[50]:


print('DecisionTreeClassifier Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test2) * 100))
#print('dtcisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, dtc_reg_predict_proba) * 100))
print('DecisionTreeClassifier Test Classification report:\n\n', classification_report(Y_test, prediction_test2))
print('DecisionTreeClassifier Training set score: {:.2f}%'.format(dtc.score(X_train_pca, Y_train) * 100))
print('DecisionTreeClassifier Testing set score: {:.2f}%'.format(dtc.score(X_test_pca, Y_test) * 100))


# In[51]:


fpr, tpr, thresholds = roc_curve(Y_test, dtc_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for DecisionTree')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[52]:


# The confusion matrix and the accuracy of the models on the test data
cm2=confusion_matrix(Y_test,prediction_test2)
  
TN = cm2[0][0]
TP = cm2[1][1]
FN = cm2[1][0]
FP = cm2[0][1]
  
print(cm2)
print()
print('Model DecisionTreeClassifier Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[53]:


sns.heatmap(cm2,annot=True)
print('LogisticRegression:')
plt.savefig('h.png')


# #  (3)  RandomForestClassifier

# In[54]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators =34, criterion = 'entropy', random_state = 42,oob_score = True,)
model3=rfc.fit(X_train_pca,Y_train)
prediction_train3=rfc.predict(X_train_pca)
prediction_test3=rfc.predict(X_test_pca)
rfc_reg_predict_proba = rfc.predict_proba(X_test_pca)[:, 1]


# In[55]:


ac3 = accuracy_score(Y_test,prediction_test3)
acc.append(ac3)


# In[56]:


print(classification_report(Y_train, prediction_train3))
print()
print(' RandomForestClassifier Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train3) * 100))


# In[57]:


print('randomforest Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test3) * 100))
#print('rfcisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, rfc_reg_predict_proba) * 100))
print('randomforest Test Classification report:\n\n', classification_report(Y_test, prediction_test3))
print('randomforest Training set score: {:.2f}%'.format(rfc.score(X_train_pca, Y_train) * 100))
print('randomforest Testing set score: {:.2f}%'.format(rfc.score(X_test_pca, Y_test) * 100))


# In[58]:


fpr, tpr, thresholds = roc_curve(Y_test, rfc_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for  RandomForest')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[59]:


# The confusion matrix and the accuracy of the models on the test data
cm3=confusion_matrix(Y_test,prediction_test3)
  
TN = cm3[0][0]
TP = cm3[1][1]
FN = cm3[1][0]
FP = cm3[0][1]
  
print(cm3)
print()
print('Model RandomForestClassifier Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[60]:


sns.heatmap(cm3,annot=True)
print(' RandomForestClassifier:')
plt.savefig('h.png')


# #  (4) KNeighborsClassifier

# In[61]:


from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 3)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=10, p=5,weights='uniform')
model4=knn.fit(X_train_pca, Y_train)
prediction_train4=knn.predict(X_train_pca)
prediction_test4=knn.predict(X_test_pca)
knn_reg_predict_proba = knn.predict_proba(X_test_pca)[:, 1]


# In[62]:


ac4 = accuracy_score(Y_test,prediction_test4)
acc.append(ac4)


# In[63]:


knn.get_params().keys()


# In[64]:


print(classification_report(Y_train, prediction_train4))
print()
print('KNeighborsClassifier Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train4) * 100))


# In[65]:


print('KNeighborsClassifier Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test4) * 100))
#print('knnisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, knn_reg_predict_proba) * 100))
print('KNeighborsClassifier Test Classification report:\n\n', classification_report(Y_test, prediction_test4))
print('KNeighborsClassifier Training set score: {:.2f}%'.format(knn.score(X_train_pca, Y_train) * 100))
print('KNeighborsClassifier Testing set score: {:.2f}%'.format(knn.score(X_test_pca, Y_test) * 100))


# In[66]:


fpr, tpr, thresholds = roc_curve(Y_test, knn_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for KNeighbors')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[67]:


# The confusion matrix and the accuracy of the models on the test data
from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(Y_test,prediction_test4)
  
TN = cm4[0][0]
TP = cm4[1][1]
FN = cm4[1][0]
FP = cm4[0][1]
  
print(cm4)
print()
print('Model KNeighborsClassifier accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[68]:


sns.heatmap(cm4,annot=True)
print('KNeighborsClassifier:')
plt.savefig('h.png')


# #  (5) svc_lin

# In[69]:


from sklearn.svm import SVC
svc_lin = SVC(kernel = 'linear', random_state = 42, C= 10, gamma= 0.01, probability = True)
model5=svc_lin.fit(X_train_pca, Y_train)
prediction_train5=svc_lin.predict(X_train_pca)
prediction_test5=svc_lin.predict(X_test_pca)
svc_lin_reg_predict_proba = svc_lin.predict_proba(X_test_pca)[:, 1]


# In[70]:


ac5 = accuracy_score(Y_test,prediction_test5)
acc.append(ac5)


# In[71]:


print(classification_report(Y_train, prediction_train5))
print()
print('Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train5) * 100))


# In[72]:


print('SVC_linear Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test5) * 100))
#print('svc_linisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, svc_lin_reg_predict_proba) * 100))
print('SVC_linear Test Classification report:\n\n', classification_report(Y_test, prediction_test5))
print('SVC_linear Training set score: {:.2f}%'.format(svc_lin.score(X_train_pca, Y_train) * 100))
print('SVC_linear Testing set score: {:.2f}%'.format(svc_lin.score(X_test_pca, Y_test) * 100))


# In[73]:


fpr, tpr, thresholds = roc_curve(Y_test, svc_lin_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for svc_lin')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[74]:


# The confusion matrix and the accuracy of the models on the test data
from sklearn.metrics import confusion_matrix
cm5=confusion_matrix(Y_test,prediction_test5)
  
TN = cm5[0][0]
TP = cm5[1][1]
FN = cm5[1][0]
FP = cm5[0][1]
  
print(cm5)
print()
print('Model SVM Linear accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[75]:


sns.heatmap(cm5,annot=True)
print('SVM Linear:')
plt.savefig('h.png')


# # (6) svc_rbf

# In[76]:


from sklearn.svm import SVC
svc_rbf = SVC(kernel = 'rbf', random_state = 10, C= 10000, gamma= 0.001,probability = True)
model6=svc_rbf.fit(X_train_pca, Y_train)
prediction_train6=svc_rbf.predict(X_train_pca)
prediction_test6=svc_rbf.predict(X_test_pca)
svc_rbf_reg_predict_proba = svc_rbf .predict_proba(X_test_pca)[:, 1]


# In[77]:


ac6 = accuracy_score(Y_test,prediction_test6)
acc.append(ac6)


# In[78]:


print(classification_report(Y_train, prediction_train6))
print()
print('Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train6) * 100))


# In[79]:


print('SVC_rbf Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test6) * 100))
#print('svc_rbfisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, svc_rbf_reg_predict_proba) * 100))
print('SVC_rbf Test Classification report:\n\n', classification_report(Y_test, prediction_test6))
print('SVC_rbf Training set score: {:.2f}%'.format(svc_rbf.score(X_train_pca, Y_train) * 100))
print('SVC_rbf Testing set score: {:.2f}%'.format(svc_rbf.score(X_test_pca, Y_test) * 100))


# In[80]:


fpr, tpr, thresholds = roc_curve(Y_test, svc_rbf_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for svc_rbf')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[81]:


# The confusion matrix and the accuracy of the models on the test data
from sklearn.metrics import confusion_matrix
cm6=confusion_matrix(Y_test,prediction_test6)
  
TN = cm6[0][0]
TP = cm6[1][1]
FN = cm6[1][0]
FP = cm6[0][1]
  
print(cm6)
print()
print('Model SVM rbf accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[82]:


sns.heatmap(cm6,annot=True)
print('SVM rbf:')
plt.savefig('h.png')


# # (7) GaussianNB

# In[83]:


from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB(var_smoothing=0)
model7=gauss.fit(X_train_pca, Y_train)
prediction_train7=gauss.predict(X_train_pca)
prediction_test7=gauss.predict(X_test_pca)
gauss_reg_predict_proba = gauss .predict_proba(X_test_pca)[:, 1]


# In[84]:


gauss.get_params().keys()


# In[85]:


ac6 = accuracy_score(Y_test,prediction_test6)
acc.append(ac6)


# In[86]:


print(classification_report(Y_train, prediction_train7))
print()
print('Training accuracy {:.2f}%'.format(accuracy_score(Y_train, prediction_train7) * 100))


# In[87]:


print('GaussianNB Testing Accuracy: {:.2f}%'.format(accuracy_score(Y_test, prediction_test7) * 100))
#print('gaussisticRegression AUC: {:.2f}%'.format(roc_auc_score(Y_test, gauss_reg_predict_proba) * 100))
print('GaussianNB Test Classification report:\n\n', classification_report(Y_test, prediction_test7))
print('GaussianNB Training set score: {:.2f}%'.format(gauss.score(X_train_pca, Y_train) * 100))
print('GaussianNB Testing set score: {:.2f}%'.format(gauss.score(X_test_pca, Y_test) * 100))


# In[88]:


fpr, tpr, thresholds = roc_curve(Y_test, gauss_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([-0.03, 1.0])
plt.ylim([0.00, 1.03])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for GaussianNB')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[89]:


# The confusion matrix and the accuracy of the models on the test data
from sklearn.metrics import confusion_matrix
cm7=confusion_matrix(Y_test,prediction_test7)
  
TN = cm7[0][0]
TP = cm7[1][1]
FN = cm7[1][0]
FP = cm7[0][1]
  
print(cm7)
print()
print('Model GaussianNB accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


# In[90]:


sns.heatmap(cm7,annot=True)
print('GaussianNB :')
plt.savefig('h.png')


# ## ROC CURVE ANALYSIS

# In[91]:


r_probs = [0 for _ in range(len(Y_test))]
log_reg_predict_proba = log.predict_proba(X_test_pca)
dtc_reg_predict_proba = dtc.predict_proba(X_test_pca)
rfc_reg_predict_proba = rfc.predict_proba(X_test_pca)
knn_reg_predict_proba = knn.predict_proba(X_test_pca)
svc_lin_reg_predict_proba = svc_lin.predict_proba(X_test_pca)
svc_rbf_reg_predict_proba = svc_rbf .predict_proba(X_test_pca)
gauss_reg_predict_proba = gauss .predict_proba(X_test_pca)


# In[92]:


log_reg_predict_proba = log_reg_predict_proba[:, 1]
dtc_reg_predict_proba = dtc_reg_predict_proba[:, 1]
rfc_reg_predict_proba = rfc_reg_predict_proba[:, 1]
knn_reg_predict_proba = knn_reg_predict_proba[:, 1]
svc_lin_reg_predict_proba = svc_lin_reg_predict_proba[:, 1]
svc_rbf_reg_predict_proba = svc_rbf_reg_predict_proba[:, 1]
gauss_reg_predict_proba = gauss_reg_predict_proba [:, 1]


# In[93]:



from sklearn.metrics import roc_curve, roc_auc_score

Calculate AUROC
ROC is the receiver operating characteristic AUROC is the area under the ROC curve
# In[94]:


r_auc = roc_auc_score(Y_test, r_probs)
log_auc = roc_auc_score(Y_test, log_reg_predict_proba)
dtc_auc = roc_auc_score(Y_test, dtc_reg_predict_proba )
rfc_auc = roc_auc_score(Y_test, rfc_reg_predict_proba )
knn_auc = roc_auc_score(Y_test, knn_reg_predict_proba)
svc_lin_auc = roc_auc_score(Y_test, svc_lin_reg_predict_proba)
svc_rbf_auc = roc_auc_score(Y_test, svc_rbf_reg_predict_proba )
gauss_auc = roc_auc_score(Y_test, gauss_reg_predict_proba)


# In[95]:


print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('LogisticRegression: AUROC = %.3f' % (log_auc))
print('DecisionTree: AUROC = %.3f' % (dtc_auc))
print('Random Forest: AUROC = %.3f' % (rfc_auc))
print('KNeighbors: AUROC = %.3f' % (knn_auc))
print('SVM_linear: AUROC = %.3f' % (svc_lin_auc))
print('SVM_rbf: AUROC = %.3f' % (svc_rbf_auc))
print('GaussianNB: AUROC = %.3f' % (gauss_auc))


# In[96]:


r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
log_fpr, log_tpr, _ = roc_curve(Y_test, log_reg_predict_proba)
dtc_fpr, dtc_tpr, _  = roc_curve(Y_test, dtc_reg_predict_proba )
rfc_fpr, rfc_tpr, _  = roc_curve(Y_test, rfc_reg_predict_proba )
knn_fpr, knn_tpr, _ = roc_curve(Y_test, knn_reg_predict_proba)
svc_lin_fpr, svc_lin_tpr, _ = roc_curve(Y_test, svc_lin_reg_predict_proba)
svc_rbf_fpr, svc_rbf_tpr, _ = roc_curve(Y_test, svc_rbf_reg_predict_proba )
gauss_fpr, gauss_tpr, _ = roc_curve(Y_test, gauss_reg_predict_proba)


# In[97]:


import matplotlib.pyplot as plt


# In[98]:


plt.plot(r_fpr, r_tpr, linestyle='--', linewidth=6, label='Random-prediction (AUROC = %0.3f)' % r_auc)
plt.plot(log_fpr, log_tpr, marker='.', linewidth=6, label='LogisticRegression (AUROC = %0.3f)' % log_auc)
plt.plot(dtc_fpr, dtc_tpr, marker='.', linewidth=6, label='DecisionTree (AUROC = %0.3f)' % dtc_auc)
plt.plot(rfc_fpr, rfc_tpr, marker='.', linewidth=6, label='Random Forest (AUROC = %0.3f)' % rfc_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', linewidth=6, label='KNeighbors (AUROC = %0.3f)' % knn_auc)
plt.plot(svc_lin_fpr, svc_lin_tpr, marker='.', linewidth=6, label='SVM_linear (AUROC = %0.3f)' % svc_lin_auc)
plt.plot(svc_rbf_fpr, svc_rbf_tpr, marker='.', linewidth=6, label='SVM_rbf (AUROC = %0.3f)' % svc_rbf_auc)
plt.plot(gauss_fpr, gauss_tpr, marker='.', linewidth=6, label='GaussianNB (AUROC = %0.3f)' % gauss_auc)

plt.rcParams['figure.figsize'] = (20,25)
plt.rcParams['font.size'] = 30

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


# # confusion matrix and the accuracy 
# 

# In[99]:


# The confusion matrix and the accuracy of the models on the test data
from sklearn.metrics import confusion_matrix
    
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,prediction_test1)
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print('Model LogisticRegression Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line



cm2=confusion_matrix(Y_test,prediction_test2)
  
TN = cm2[0][0]
TP = cm2[1][1]
FN = cm2[1][0]
FP = cm2[0][1]
  
print(cm2)
print('Model DecisionTreeClassifier Testing accuracy =',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line



cm3=confusion_matrix(Y_test,prediction_test3)
  
TN = cm3[0][0]
TP = cm3[1][1]
FN = cm3[1][0]
FP = cm3[0][1]
  
print(cm3)
print()
print('Model RandomForest Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line


cm4=confusion_matrix(Y_test,prediction_test4)
  
TN = cm4[0][0]
TP = cm4[1][1]
FN = cm4[1][0]
FP = cm4[0][1]
  
print(cm4)
print()
print('Model KNeighborsClassifier Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line



cm5=confusion_matrix(Y_test,prediction_test5)
  
TN = cm5[0][0]
TP = cm5[1][1]
FN = cm5[1][0]
FP = cm5[0][1]
  
print(cm5)
print('Model SVM Linear Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line



cm6=confusion_matrix(Y_test,prediction_test6)
  
TN = cm6[0][0]
TP = cm6[1][1]
FN = cm6[1][0]
FP = cm6[0][1]
  
print(cm6)
print('Model SVM rbf Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line



cm7=confusion_matrix(Y_test,prediction_test7)
  
TN = cm7[0][0]
TP = cm7[1][1]
FN = cm7[1][0]
FP = cm7[0][1]
  
print(cm7)
print('Model GaussianNB Testing accuracy = ',((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line
print()# Print a new line


# #  Training accuracy
# 

# In[100]:


#print model accuracy on the training data.
print('[1]Logistic Regression Training accuracy: {:.2f}%'.format(log.score(X_train_pca, Y_train)*100))
print()
print('[2]Decision Tree Classifier Training accuracy: {:.2f}%'.format(dtc.score(X_train_pca, Y_train)*100))
print()
print('[3]Random Forest Classifier Training accuracy: {:.2f}%'.format(rfc.score(X_train_pca, Y_train)*100))
print()
print('[4]K Nearest Neighbor Training accuracy: {:.2f}%'.format(knn.score(X_train_pca, Y_train)*100))
print()
print('[5]Support Vector Machine (Linear Classifier) Training accuracy: {:.2f}%'.format(svc_lin.score(X_train_pca, Y_train)*100))
print()
print('[6]Support Vector Machine (RBF Classifier) Training accuracy: {:.2f}%'.format(svc_rbf.score(X_train_pca, Y_train)*100))
print()
print('[7]Gaussian Naive Bayes Training accuracy: {:.2f}%'.format(gauss.score(X_train_pca, Y_train)*100))


# # Testing accuracy 

# In[101]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print('LogisticRegression testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test1) * 100))
print()
print('DecisionTreeClassifier testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test2) * 100))
print()
print('RandomForestClassifier testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test3) * 100))
print()
print('KNeighborsClassifier testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test4) * 100))
print()
print('SVM Linear testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test5) * 100))
print()
print('SVM rbf yesting accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test6) * 100))
print()
print('GaussianNB testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test7) * 100))
print()


# In[102]:


#random forest
print('SVC_rbc testing accuracy_score : {:.2f}%'.format(accuracy_score(Y_test, prediction_test6) * 100))

print()

pred = model6.predict(X_test_pca)
print(pred)

#Print a space
print()

#Print the actual values
print(Y_test)

