#firstly import all important laibararies
from tkinter import X
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#App Heading
st.write(""" 
         # Explore diffrent ML Models and datasets
          Check which model is best from all of them""")

#put Dataset name in a box and put them  on sidebar
Dataset_name = st.sidebar.selectbox(
    'Slect Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

#or put classifier names in box behind them
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#Define a function to load dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
        
    x = data.data
    y = data.target
    return x,y 

#know call above function and take it equall x and y variable
x, y = get_dataset(Dataset_name)

#know we print shape of our dataset on app
st.write('Shape of Dataset: ', x.shape)
st.write('Number of classes : ',len(np.unique(y)))

#Next we use different classifiers parameters and add them in input
def add_parameter_ui(classifier_name):
    params = dict()  #crate an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C',0.01, 10.0)
        params['C'] = C #its the degree of corect classifier
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K',1, 15)
        params['K'] = K  # its the number of nearest neibhours
    else:
        max_depth = st.sidebar('max_depth',2,15)
        params['max_depth'] = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth #depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators #number of trees
    return params 

# Call Above function and take it Equall toparams variable
params = add_parameter_ui(classifier_name)

#know we make a classifier base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C =params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],random_state=1234)
    return clf
    
#Know call above function and take it equall to clf variable
clf = get_classifier(classifier_name,params)

#know split our data into test and train
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1234)

#know we train our classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

#Know check accuracy score and print it on app
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = { classifier_name}')
st.write(f'Accuracy = ', acc)


### Plot Dataset###
#know we convert our all features into 2-d plot by using PCA
pca = PCA(2) # convert our multi dimentional plot into 2-d plot
X_projected = pca.fit_transform(x)

#know we convert our data into 0 and 1 slice
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y,alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)