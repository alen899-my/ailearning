# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#create synthtetic dataset
from sklearn.datasets import make_classification

# %%
X,y=make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0)

# %%
X

# %%
y
pd.DataFrame(X)[0]

# %%
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Label'] = y

# 3️⃣ Plot with seaborn
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Label', palette='Set1')
plt.show()

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# %%


# %%


# %%


# %%
from sklearn.svm import SVC

# %%
svc=SVC(kernel='linear')

# %%
svc.fit(X_train,y_train)

# %%
y_pred=svc.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# %%
print(classification_report(y_test,y_pred))

# %%
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# %%


# %%
rbf=SVC(kernel="rbf")

# %%
rbf.fit(X_train,y_train)

# %%
y_pred1=rbf.predict(X_test)

# %%
print(classification_report(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))

# %%
poly=SVC(kernel="poly")

# %%
poly.fit(X_train,y_train)

# %%
y_pred2=rbf.predict(X_test)

# %%
print(classification_report(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))


