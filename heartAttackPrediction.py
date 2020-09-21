import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 300)
pd.set_option('display.max_column', 15)

# introduction
print(" NAME : Hrishikesh M Thakur \n BATCH : Machine Learning \n TOPIC : Creating the modal for identifying patients likely to get the Heart attack \n Email : hrishith27@gmail.com")

# read data form the file
df = pd.read_csv('HeartAttack_data.csv')

# analysing the dataset provided
print("\n\n\n----------------------------Analysing the data set provided----------------------------------- \n", df.shape)
print(df.describe())

# data processing
df.drop(["slope", "ca", "thal"], axis=1, inplace=True)
imputer = SimpleImputer(missing_values='?', strategy="constant", fill_value=np.nan)
imputer.fit(df)
SimpleImputer()
col = df.columns.values
df = imputer.transform(df)
df = pd.DataFrame(df, columns=col)
df = df.astype(float)
df["num"] = df.num.astype(int)
df.fillna(df.median(), inplace=True)

# data frame after processing the data
print("\n\n\n----------------------------DataFrame after processing the the data----------------------------------\n\n")
print(df)

# analysis for predicting the modal
print("\n\n\n---------Analysis Predicting the model------------\n")
print(df.groupby('num').size())
plt.hist(df['num'])
plt.title('HEART ATTACK \n(Prone to Heart Attack=1 , Not prone to Heart Attack=0)')
plt.show()
df.plot(kind='density', subplots=True, layout=(4, 3), sharex=False, legend=False, fontsize=1)
plt.title("Density plot of each column")
plt.show()

# Correlation of attributes
fig=plt.figure()
ax1=fig.add_subplot(111)
cax=ax1.imshow(df.corr())
ax1.grid(True)
print(plt.title('Heart Attack Attributes Correlation'))
fig.colorbar(cax, ticks=[.5, .6, .7, .8, .9, 1])
print(plt.show())

# Split train and test data set data
Y = df.num.values
X = df.drop("num",axis=1).values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=21)
List=[]
List.append(('CART', DecisionTreeClassifier()))
List.append(('SVM', SVC()))
List.append(('NB', GaussianNB()))
List.append(('KNN', KNeighborsClassifier()))
num_folds = 12
results = []
names = []
print("\n\n\nAccuracies of algorithm \n")
for name, model in List:
    kfold = KFold(n_splits=num_folds, random_state=123)
    startTime = time.time()
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "{}: {} ({}) (run time: {})".format(name, cv_results.mean(), cv_results.std(), endTime-startTime))


# Performance Comparision
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))


results = []
names = []



print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=238)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print("{}: {} ({}) (run time: {})" .format(name, cv_results.mean(), cv_results.std(), end-start))


fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




# prepare the model

scaler = StandardScaler().fit(x_train)

X_train_scaled = scaler.transform(x_train)
model = GaussianNB()
start = time.time()
model.fit(X_train_scaled, y_train)   #Training of algorithm
end = time.time()
print("\n\nGaussian NB Training Completed. It's Run Time: %f" % (end-start))
x_test_scaled = scaler.transform(x_test)
predictions = model.predict(x_test_scaled)
print("All predictions done successfully by NB Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(y_test, predictions))
print("confusion_matrix =")
print( confusion_matrix(y_test, predictions))
from sklearn.externals import joblib
filename =  "finalized_heartAttack_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")

print(" NAME : Hrishikesh M Thakur \n BATCH : Machine Learning \n TOPIC : Creating the modal for identifying patients likely to get the Heart attack \n Email : hrishith27@gmail.com")
