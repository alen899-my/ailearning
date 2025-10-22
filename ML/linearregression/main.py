import numpy as np
import pandas as pd

#seeding
np.random.seed(20)

#number of samples
n_samples=1000
class_zero_ratio=0.9

n_class_zero=int(n_samples*class_zero_ratio)
n_class_one=n_samples-n_class_zero


#generate feaetures for zero
feature1_class_zero=np.random.normal(loc=0,scale=1,size=n_class_zero)
feature2_class_zero=np.random.normal(loc=0,scale=1,size=n_class_zero)

#generate feature for one
feature1_class_one=np.random.normal(loc=1,scale=1,size=n_class_one)
feature2_class_one=np.random.normal(loc=1,scale=1,size=n_class_one)

#create dataframes

class_zero_df=pd.DataFrame({
    'feature1':feature1_class_one,
    "feature2":feature2_class_one,
    'target':0
})

class_one_df=pd.DataFrame({
    'feature1':feature1_class_zero,
    "feature2":feature2_class_zero,
    'target':1
})

df=pd.concat([class_zero_df,class_one_df],ignore_index=True)

value_count=df["target"].value_counts()
print(value_count)
from sklearn.utils import resample