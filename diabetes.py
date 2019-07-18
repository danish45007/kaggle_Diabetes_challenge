import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('diabetes.csv')

#To cheack if any null values are present
'''print(dataset.isnull().values.any())'''

#correlation

corr = dataset.corr()
#plot using heatmaps
top_corr = corr.index

plt.figure(figsize=(20,20))

g = sns.heatmap(dataset[top_corr].corr(),annot=True,cmap=('RdYlGn'))

'''plt.show(g)'''

#changing the diabetes column data from boolean to nuumber(0,1)

'''data = {True: 1, False:0}

dataset['diabetes'] = dataset['diabetes'].map(data)

print(dataset.head(5))'''

#true and false count

data_true_count = len(dataset.loc[dataset['Outcome']==True])

data_false_count = len(dataset.loc[dataset['Outcome']==False])

print(data_true_count,data_false_count)

#Train and test


from sklearn.model_selection import train_test_split

feature_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

predict_colum = ['Outcome']

x = dataset[feature_columns].values

y = dataset[predict_colum].values

X_train,X_test,y_trian,y_test = train_test_split(x ,y, test_size =0.30,random_state=10)

#to fill the rows having zero values with mean

from sklearn.preprocessing import Imputer
fill_values  = Imputer(missing_values=0,strategy='mean',axis=0)

X_train = fill_values.fit_transform(X_train)

y_trian = fill_values.fit_transform(y_trian)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=10)

random_forest.fit(X_train,y_trian.ravel())

predict = random_forest.predict(X_test)

from sklearn import metrics
print("Accuracy  = {0:3f}".format(metrics.accuracy_score(y_test,predict)))




