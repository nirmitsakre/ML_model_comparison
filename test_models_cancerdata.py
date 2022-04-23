#visualization packages
import matplotlib.pyplot as plt
#model packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#dataset package
from sklearn.datasets import load_breast_cancer
#split dataset package
from sklearn.model_selection import train_test_split
#roc package
from sklearn.metrics import roc_curve

#load dataset
Data_C = load_breast_cancer()

#split into x (features) and y (target)
x = Data_C.data
y = Data_C.target

#train and test dataset split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,train_size=0.7, random_state= 88)

#Lr
Lr_canc = LogisticRegression()
Lr_canc.fit(X_train,y_train)
y_prob_Lr = Lr_canc.predict_proba(X_test)
y_prob_Lr = y_prob_Lr[:,1]
FPR_Lr, TPR_Lr, Thresholds_Lr = roc_curve(y_test, y_prob_Lr)

#Dt
Dt_canc = DecisionTreeClassifier()
Dt_canc.fit(X_train,y_train)
y_prob_Dt = Dt_canc.predict_proba(X_test)
y_prob_Dt = y_prob_Dt[:,1]
FPR_Dt, TPR_Dt, Thresholds_Dt = roc_curve(y_test, y_prob_Dt)

#NB
NB_canc = GaussianNB()
NB_canc.fit(X_train,y_train)
y_prob_NB = NB_canc.predict_proba(X_test)
y_prob_NB = y_prob_NB[:,1]
FPR_NB, TPR_NB, Thresholds_NB = roc_curve(y_test, y_prob_NB)

#kNN
kNN_canc = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
kNN_canc.fit(X_train,y_train)
y_prob_kNN = kNN_canc.predict_proba(X_test)
y_prob_kNN = y_prob_kNN[:,1]
FPR_kNN, TPR_kNN, Thresholds_kNN = roc_curve(y_test, y_prob_kNN)

#Plot ROC
plt.plot(FPR_Lr,TPR_Lr,label="LR")
plt.plot(FPR_Dt,TPR_Dt,label="DT")
plt.plot(FPR_NB,TPR_NB,label="NB")
plt.plot(FPR_kNN,TPR_kNN,label="kNN")
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
