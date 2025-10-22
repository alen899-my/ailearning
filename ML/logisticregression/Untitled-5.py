# %%
import seaborn as sns
data=sns.load_dataset("tips")

# %%
data.head()

# %%
data.info()

# %%
data["sex"].value_counts()

# %%
data["smoker"].value_counts()

# %%
data["day"].value_counts()

# %%
data["time"].value_counts()

# %%
#feature encoding[label][onehot]

# %%
#indipendend and dependede features
data.columns

# %%


# %%
X=data[[ 'tip', 'sex', 'smoker', 'day', 'time', 'size']]

# %%
y=data["total_bill"]

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# %%
X_train

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()

# %%
X_train["sex"]=le1.fit_transform(X_train['sex'])
X_train["smoker"]=le2.fit_transform(X_train["smoker"])
X_train["time"]=le3.fit_transform(X_train["time"])

# %%


# %%


# %%
X_train

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
X_test["sex"]=le1.transform(X_test['sex'])
X_test["smoker"]=le2.transform(X_test["smoker"])
X_test["time"]=le3.transform(X_test["time"])

# %%
X_test.head()

# %%
#onehot encoding---column transformer
from sklearn.compose import ColumnTransformer

# %%
from sklearn.preprocessing import OneHotEncoder

# %%
ct=ColumnTransformer(transformers=[('onehot',OneHotEncoder(drop='first'),[3])],remainder="passthrough")

# %%
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
X_train=ct.fit_transform(X_train)

# %%
X_test=ct.transform(X_test)

# %%
X_test

# %%
#support vector regrssion
from sklearn.svm import SVR
svr=SVR()

# %%
svr.fit(X_train,y_train)

# %%
y_pred=svr.predict(X_test)

# %%
from sklearn.metrics import r2_score ,mean_absolute_error

# %%
r2_score(y_test,y_pred)

# %%
#tuning
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# %%
param_grid={'C': [0.1,.1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001,0.0001],
    'kernel':['rbf']

}

# %%
grid=GridSearchCV(SVC(),param_grid=param_grid,cv=5,verbose=3,refit=True)

# %%
grid.fit(X_train,y_train)

# %%
grid.best_params_

# %%


# %%
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVR(), param_grid=param_grid, cv=5, verbose=3, refit=True)
grid.fit(X_train, y_train)  # now y_train is continuous, so this works correctly


# %%
grid.best_params_

# %%
grid_pred=grid.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
print(accuracy_score(y_test,grid_pred))


