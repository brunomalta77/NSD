from PIL import Image
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import pandas as pd 


# getting the lotties request
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#load assets
lottie_coding_normal =load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_E94I4T.json")
lottie_coding_ligeiro =load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_GzYRQ3.json")
lottie_coding_moderado =load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_zC6KybFGMj.json")
lottie_coding_grave =load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_DcEAr9.json")


#------loading the models-------
scaler_loaded = joblib.load(r'standard_scaler.pkl')
RF_loaded = joblib.load(r'random_forest_model.pkl')

# page config 
st.set_page_config(page_title="My_App",page_icon="💤",layout="wide")

#Explain what we are going to do
with st.container():
    st.header("NSD V1.0 (Nox Saos Detect)")
    st.write("Using an algorithm to assess and automatize the classification of SAOS after using a NOXT3s")

with st.container():
    st.write("""In a time where the importance of sleep is enormous and the evolution of IA is very proeminent, our main goal is to try
    to try to massificate the diagnosis of saos, using NOXT3s, for increasing the accuracy of the classification.""")

def my_values():
    age= int(st.number_input("Tell me the age",1))
    weight = float(st.number_input("Tell me the weight in kg",1))
    height = float(st.number_input("Tell me the height in cm",1))
    IMC = round(weight/(height**2),1)
    IAH_Auto = float(st.number_input("Tell me the IAH_Auto",1))
    ODI_Auto = float(st.number_input("Tell me the ODI_Auto",1))
    EP = int(st.number_input("Tell me the Epworth scale",1))
    Cervical = float(st.number_input("Tell me the cervical perimeter in cm",1))
    Abdominal = float(st.number_input("Tell me the abdominal perimeter in cm",1))
    original_df = pd.DataFrame({"Idade":age,"Peso":weight,"Altura":height,"IMC":IMC,"IAH-Auto":IAH_Auto,"ODI-Auto":ODI_Auto,"E.EP":EP,"P.Cervical":Cervical,"P.Abdominal":Abdominal},index=[0])
    transform_df= scaler_loaded.transform(original_df)
    Genre = st.selectbox("Select the Genre",("M", "F"))
    if Genre == "M":
        Sexo = pd.DataFrame({"F":0,"M":1},index=[0])
    else:
        Sexo = pd.DataFrame({"F":1,"M":0},index=[0])

    df_final = pd.concat([pd.DataFrame(transform_df, columns= original_df.columns),Sexo],axis=1)
    return df_final

#row= my_values()

###classification###

def classification(model,row):
    res = model.predict(row)
    if res == [0]:
        st.write(f'Normal')
        st_lottie(lottie_coding_normal, height = 300,key="coding")
    if res == [1]:
         st.write(f'Ligeiro')
         st_lottie(lottie_coding_ligeiro, height = 300,key="coding")
    if res == [2]:
         st.write(f'Moderado')
         st_lottie(lottie_coding_moderado, height = 300,key="coding")
    if res == [3]:
         st.write(f'Grave')
         st_lottie(lottie_coding_grave, height = 300,key="coding")
    return res 

#classification(RF_loaded,row)
def main():
    #global variables
    
    
    
    with st.container():
        left_column,right_column = st.columns(2)
        with left_column:
            row = my_values()
            st.write("To generate your IAH classification, click on the button")
            if st.button("Generate IAH classification"):
                classification(RF_loaded,row)



if __name__=="__main__":
    main()
