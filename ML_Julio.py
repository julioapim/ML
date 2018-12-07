import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier

data_train = pd.read_csv('train_n2.csv')
data_test = pd.read_csv('test_n2.csv')


"""# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  """

#############Ate aqui

#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

#Dummies Variables
embarked_train = pd.get_dummies( data_train.Embarked , prefix='Embarked' )
embarked_test = pd.get_dummies( data_test.Embarked , prefix='Embarked' )

pclass_train = pd.get_dummies( data_train.Pclass , prefix='Pclass' )
pclass_test = pd.get_dummies( data_test.Pclass , prefix='Pclass' )

#Concat train Variables
X = pd.concat( [ pclass_train , data_train.Sex , data_train.Age , 
                             data_train.SibSp , data_train.Parch , embarked_train , data_train.Fare ] , axis=1 )
y = data_train["Survived"].values

#Test Variables
X_prod = pd.concat( [ pclass_test , data_test.Sex , data_test.Age , 
                             data_test.SibSp , data_test.Parch , embarked_test , data_test.Fare ] , axis=1 )

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)











# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
Xgboost = XGBClassifier()
Xgboost.fit(X_train, y_train)

#SVM incial
clf = svm.SVC(gamma=0.001, C=100.)
my_svn_one = clf.fit(X_train, y_train)
print("Score SVC:", my_svn_one.score(X_train, y_train))

########################Test Pack########################
#PassengerId = np.array(data_test["PassengerId"]).astype(int)
#solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
#solucao.to_csv("solucao3.csv", index_label = ["PassengerId"])
#print(solucao.shape)


#RandomForestClassifier
Model_RFC = RandomForestClassifier(n_estimators=100)
RFC = Model_RFC.fit(X_train, y_train)
print("Score RFC:", Model_RFC.score(X_train, y_train))

########################Test Pack########################
#target_test = RFC.predict(features_test)
#PassengerId = np.array(data_test["PassengerId"]).astype(int)
#solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
#solucao.to_csv("solucao4.csv", index_label = ["PassengerId"])
#print(solucao.shape)


#Gradient Boosting Classifier
Model_GBC = GradientBoostingClassifier()
GBC = Model_GBC.fit(X_train, y_train)
print("Score_GBC:", Model_GBC.score(X_train, y_train))

########################Test Pack########################
#target_test = GBC.predict(features_test)
#PassengerId = np.array(data_test["PassengerId"]).astype(int)
#solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
#solucao.to_csv("solucao5.csv", index_label = ["PassengerId"])
#print(solucao.shape)

#K-nearest neighbors
Model_KNC = KNeighborsClassifier(n_neighbors = 3)
KNC = Model_KNC.fit(X_train, y_train)
print("Score_KNC:", Model_KNC.score(X_train, y_train))

########################Test Pack########################
#target_test = KNC.predict(features_test)
#PassengerId = np.array(data_test["PassengerId"]).astype(int)
#solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
#solucao.to_csv("solucao6.csv", index_label = ["PassengerId"])
#print(solucao.shape)

#Gaussian Naive Bayes
Model_GNB = GaussianNB()
GNB = Model_GNB.fit(X_train, y_train)
print("Score_GNB:", Model_GNB.score(X_train, y_train))

########################Test Pack########################
#target_test = GNB.predict(features_test)
#PassengerId = np.array(data_test["PassengerId"]).astype(int)
#solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
#solucao.to_csv("solucao7.csv", index_label = ["PassengerId"])
#print(solucao.shape)


#Logistic Regression
Model_LogR = LogisticRegression()
LogR = Model_LogR.fit(X_train, y_train)
print("Score_LogR:", Model_LogR.score(X_train, y_train))


#Final
# Predicting the Test set results
y_pred = Xgboost.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


########################Test Pack########################

target_test = Xgboost.predict(X_prod)
PassengerId = np.array(data_test["PassengerId"]).astype(int)
solucao = pd.DataFrame(target_test, PassengerId, columns = ['Survived'])
solucao.to_csv("solucao19.csv", index_label = ["PassengerId"])
print(solucao.shape)



#Analise
"""# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
Xgboost_CV = cross_val_score(estimator = Xgboost, X = X_train, y = y_train, cv = 10)
Xgboost_CV.mean()
Xgboost_CV.std()"""
