import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from collinearity import SelectNonCollinear
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

#Seperate Feature

data = pd.read_csv('wafer_data.csv')

y = data["Class"]
x = data.drop(['Class'], axis = 1)

#StandardScaler
SS = StandardScaler()
SS.fit(x)
X_SS = SS.transform(x)
X_SS_PD = pd.DataFrame(X_SS)

#MaxAbsScaler
MAS = MaxAbsScaler()
MAS.fit(x)
X_MAS = MAS.transform(x)

#yeo-johnson Transformation
PTF=PowerTransformer()
PTF.fit(X_MAS)
X_PTF = PTF.transform(X_MAS)

#remove_multicollinearity
selector = SelectNonCollinear(0.90)
selector.fit(X_PTF,y)
X_sel=selector.transform(X_PTF)

#feature_selection 
VT = VarianceThreshold(threshold=(.3 * (1 - .3)))
XXXX=VT.fit_transform(X_sel)

#unknown_categorical
Xi = SimpleImputer(strategy='mean')
XX=Xi.fit_transform(XXXX) 
#most_frequent


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(XX, y, test_size=0.2, random_state=10)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(XX, y, test_size=0.2, random_state=20)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(XX, y, test_size=0.2, random_state=30)
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(XX, y, test_size=0.2, random_state=40)
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(XX, y, test_size=0.2, random_state=50)
X_train6, X_test6, Y_train6, Y_test6 = train_test_split(XX, y, test_size=0.2, random_state=60)
X_train7, X_test7, Y_train7, Y_test7 = train_test_split(XX, y, test_size=0.2, random_state=70)
X_train8, X_test8, Y_train8, Y_test8 = train_test_split(XX, y, test_size=0.2, random_state=80)
X_train9, X_test9, Y_train9, Y_test9 = train_test_split(XX, y, test_size=0.2, random_state=90)
X_train10, X_test10, Y_train10, Y_test10 = train_test_split(XX, y, test_size=0.2, random_state=100)

#fix_imbalance
smote = SMOTE(random_state=10)
X_train_over1,y_train_over1 = smote.fit_resample(X_train1,Y_train1)
X_train_over2,y_train_over2 = smote.fit_resample(X_train2,Y_train2)
X_train_over3,y_train_over3 = smote.fit_resample(X_train3,Y_train3)
X_train_over4,y_train_over4 = smote.fit_resample(X_train4,Y_train4)
X_train_over5,y_train_over5 = smote.fit_resample(X_train5,Y_train5)
X_train_over6,y_train_over6 = smote.fit_resample(X_train6,Y_train6)
X_train_over7,y_train_over7 = smote.fit_resample(X_train7,Y_train7)
X_train_over8,y_train_over8 = smote.fit_resample(X_train8,Y_train8)
X_train_over9,y_train_over9 = smote.fit_resample(X_train9,Y_train9)
X_train_over10,y_train_over10 = smote.fit_resample(X_train10,Y_train10)
#rf = LGBMClassifier(n_estimators=100,min_split_gain=0.9,min_child_samples=21,learning_rate=0.01,num_leaves=40,reg_alpha=0.005,reg_lambda=0.0005,subsample_for_bin=200000,feature_fraction=0.6,bagging_freq=6,bagging_fraction=0.6)

#min_split_gain=0.9,min_child_samples=21,learning_rate=0.01,num_leaves=40,reg_alpha=0.005,reg_lambda=0.0005,subsample_for_bin=200000,feature_fraction=0.6,bagging_freq=6,bagging_fraction=0.6
rf= RandomForestClassifier(bootstrap=False, n_estimators=120,max_features='sqrt',min_impurity_decrease =0.0005)
#criterion='entropy'
#max_depth=10
#max_features='sqrt',min_samples_leaf=4,min_samples_split=2,,min_impurity_decrease =0.0005
rf.fit(X_train_over1,y_train_over1)
rf.fit(X_train_over2,y_train_over2)
rf.fit(X_train_over3,y_train_over3)
rf.fit(X_train_over4,y_train_over4)
rf.fit(X_train_over5,y_train_over5)
rf.fit(X_train_over6,y_train_over6)
rf.fit(X_train_over7,y_train_over7)
rf.fit(X_train_over8,y_train_over8)
rf.fit(X_train_over9,y_train_over9)
rf.fit(X_train_over10,y_train_over10)

# Accuracy Check!!
pred1 = rf.predict(X_test1)
pred2 = rf.predict(X_test2)
pred3 = rf.predict(X_test3)
pred4 = rf.predict(X_test4)
pred5 = rf.predict(X_test5)
pred6 = rf.predict(X_test6)
pred7 = rf.predict(X_test7)
pred8 = rf.predict(X_test8)
pred9 = rf.predict(X_test9)
pred10 = rf.predict(X_test10)

# pred_t = rf.predict(X_train_over)

# print(accuracy_score(Y_test, pred))
# print(accuracy_score(y_train_over,pred_t))


# print(confusion_matrix(y_train_over, pred_t))

AUC_AVG=roc_auc_score(Y_test1, pred1)+roc_auc_score(Y_test2, pred2)+roc_auc_score(Y_test3, pred3)+roc_auc_score(Y_test4, pred4)+roc_auc_score(Y_test5, pred5)+roc_auc_score(Y_test6, pred6)+roc_auc_score(Y_test7, pred7)+roc_auc_score(Y_test8, pred8)+roc_auc_score(Y_test9, pred9)+roc_auc_score(Y_test10, pred10)
print(AUC_AVG/10)


#sns.boxplot(data=X_SS)
#plt.show()

'''
#Use VAL_Score
log_reg_kf = LogisticRegression(random_state=13, solver = 'liblinear')
#log_reg_kf = RandomForestClassifier(n_estimators=2000)
skfold = StratifiedKFold(n_splits = 5)

score = cross_val_score(log_reg_kf, X_train, Y_train,scoring='accuracy', cv = 5)
print(np.mean(score))

with open('model_face.pkl','wb') as f:
    pickle.dump(rf,f,protocol=2)
'''
