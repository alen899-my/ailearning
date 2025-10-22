# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
%matplotlib inline

# %%
data=pd.read_csv("Football_Dataset_2015_2025.csv")
data.head()
data["Competition"].unique()

# %%
data.columns

# %%
data.isnull().sum()

# %%
from sklearn.model_selection import train_test_split


# %%
data=data.drop(["Date"],axis=1)

# %%
from sklearn.preprocessing import LabelEncoder,StandardScaler

# %%
labels_Cols=["Competition","Home Team","Away Team","Winner"]
le=LabelEncoder()

# %%
for col in labels_Cols:
    data[col]=le.fit_transform(data[col])

# %%
data.head()

# %%
data["Competition"].unique()

# %%
X=data.drop(["Winner","Year"],axis=1)
y=data["Winner"]

# %%
X

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
num_cols = ['Home Goals', 'Away Goals', 'Possession % (Home)', 'Possession % (Away)', 
            'Shots (Home)', 'Shots (Away)', 'Corners (Home)', 'Corners (Away)', 'Fouls (Home)', 'Fouls (Away)']

# %%
scaler=StandardScaler()

# %%
X_train[num_cols]=scaler.fit_transform(X_train[num_cols])

# %%
X_test[num_cols]=scaler.transform(X_test[num_cols])

# %%
from sklearn.linear_model import LogisticRegression

# %%
model=LogisticRegression(max_iter=200)

# %%
model.fit(X_train,y_train)

# %%
y_pred=model.predict(X_test)

# %%
y.unique()

# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%


# %%


