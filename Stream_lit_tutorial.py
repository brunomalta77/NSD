#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from PIL import Image
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from sklearn import datasets
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#---- Load Assets------
lottie_coding =load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_w98viu8m.json")
img_1 = Image.open("C:\\Users\\Bruno\\Research\\Images\\Figure_27.jpg.png")
img_2 = Image.open("C:\\Users\\Bruno\\Research\\Images\\Figure_27.jpg.png")

st.set_page_config(page_title="My Webpage",page_icon=":tada:",layout="wide")

#-----------Header section-----------
# using with st.container():, is important for organizing the chunks of coding. 
with st.container():
    st.subheader("Hi,I am Bruno :wave:")
    st.title("I am a Data Scientist in the making")
    st.write("i like astronomy and science")
    st.write("[astronomy >] https://www.nasa.gov/")


with st.container():
    st.write("---")
    st.write("[if you wnat to change your emojis >] (https://www.webfx.com/tools/emoji-cheat-sheet/)")

st.write("explore different datas")

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine dataset"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","RF"))

# creating the data set
def get_dataset(dataset_name):
    if dataset_name== "Iris":
        data = datasets.load_iris()
    elif dataset_name =="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)

st.write("shape of the data->",X.shape)
st.write("number of classes ->",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name=="KNN":
        K = st.sidebar.slider("K",1,15) # the start value is 1 and the stop 15. 
        params["K"] = K
    return params 


params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name =="KNN":
        clf= KNeighborsClassifier(n_neighbors=params["K"])
    return clf


clf = get_classifier(classifier_name,params)


#classification

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


acc = accuracy_score(y_test,y_pred)
st.write(f"classifier = {clf}")
st.write(f"accuracy = {acc}")

# Plotting
pca = PCA(2) # número de dimensões que queremos neste caso são 2
X_projected = pca.fit_transform(X) #unsupervised



x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal component1")
plt.ylabel("Principal component2")
plt.colorbar()

st.pyplot(fig) # para mostrar na app

#----------What i do----------------------
with st.container():
    st.write("---")
    left_column,right_column = st.columns(2)
    with left_column:
        st.header("what i do")
        st.write("##")
        st.write("Let's Gooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
        st.write("[Youtube >] (https://www.youtube.com/watch?v=_s0bcrHO0Nk)")
    with right_column:
        st_lottie(lottie_coding, height = 300,key="coding")

#-------------------------- Projects---------------------------
with st.container():
    st.write("----")
    st.header("My Projects")
    st.write("##")
    image_column,text_column = st.columns((1,2))  # neste caso a text column vai ser o dobro da segunda
    with image_column:
        st.image(img_1)
    with text_column:
        st.subheader("isso tudo")
        st.write("alsdasfasd")
        st.markdown("[Watch Video....>](https://www.youtube.com/watch?v=VqgUkExPvLY)")


with st.container():
    st.write("---")
    st.header("For changing the theme!")
    st.write("##")
    st.write("[See here >](https://www.youtube.com/watch?v=VqgUkExPvLY)")

