#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import pydeck as pdk
from streamlit_folium import folium_static
import folium
from PIL import Image,ImageEnhance
import cv2
import base64
import os
import joblib


def pages():
    st.sidebar.title('Streamlit Tools')
    selected_page = st.sidebar.selectbox('Select Application',["Main Page","ML & DL Applications","Sidebar","Widgets","Charts","Maps","Media"])
    return selected_page   
    
def main_page():
    st.title('Hello, *Streamlit!* 👨‍💻')

def ml_dl_apps():
    selected_ml_dl_app = st.sidebar.radio("Select Application",('Regression','Classification','Sentiment Analysis','Object Detection'))
    
    if selected_ml_dl_app == 'Regression':
        with st.echo(code_location='below'):
            st.write('Regression App')
            st.title("Makine Ogrenmesi ile Emlak Tahmini")
            
            mahalle_dizi = ['Cumhuriyet','Ataturk','Yildirim','Selvilitepe',
            'Subasi','Turgutlar','Ozyurt','Ergenekon',
            'Yeni','Yigitler','Acarlar','Albayrak',
            'Kurtulus','Mustafa Kemal','Sehitler','Yedi Eylul'
            'Istiklal','Dalbahce','Yilmazlar']
            isitma_dizi = ["Kombi Dogalgaz","Klima","Soba","Merkezi"]
            
            def linear_search(alist, key):
                for i in range(len(alist)):
                    if alist[i] == key:
                        return i
                return -1
            
            mahalle = st.selectbox("Mahalle",mahalle_dizi)
            mahalle_index = linear_search(mahalle_dizi,mahalle) + 1
            metrekare = st.slider("Metrekare(net)",50,200)
            oda = st.slider("Oda Sayisi(+1)",1,5)
            isitma = st.selectbox("Isitma Tipi",["Kombi Dogalgaz","Klima","Soba","Merkezi"])
            isitma_index = linear_search(isitma_dizi,isitma) + 1
            bina_Yasi = st.slider("Bina Yasi",0,30)
            bulundugu_Kat = st.slider("Bulundugu Kat",0,15)
            toplam_Kat = st.slider("Bina Toplam Kat",0,15)
            banyo_sayi = st.radio("Banyo Sayisi",('1','2'))
            esya_durum = st.radio("Esya Durum(Esyali:1 Esyasiz:0)",('0','1'))
            
            columns_model = joblib.load('regression_app/model_columns.pkl')
            
            res = pd.DataFrame(columns=columns_model,data = 
            {'Banyo_Sayisi':[1],'Bina_KatSayisi':[4],'Bina_Yasi':[0],
             'Bulundugu_Kat':[4],'Esya_Durumu':[0],'Isitma_Tipi':[isitma_index],
              'Mahalle':[mahalle_index],'Metrekare(net)':[metrekare],'Oda_Sayisi(+1)':[oda]})

            model = st.selectbox("Model",["Desicion Tree","Random Forest","SVR"])
            st.write(" Model :",model)
            
            if model == 'Desicion Tree':
                dt = joblib.load('regression_app/desicion_tree_model.pkl')
                prediction = dt.predict(res)
            
            elif model == 'Random Forest':
                rf = joblib.load('regression_app/random_forest_model.pkl')
                prediction = rf.predict(res)

            elif model == 'SVR':
                rf = joblib.load('regression_app/svr_model.pkl')
                prediction = rf.predict(res)
            
            prediction = np.round(prediction)
            tahmin_sonuc = str(prediction).strip('[]')

            if st.button("Tahmin Et"):
                st.write("Emlak Degeri :",tahmin_sonuc,"000 TL")
                st.balloons()

    elif selected_ml_dl_app == 'Classification':
        with st.echo(code_location='below'):
            st.write('Classification App')
            st.title("Wine Quality Prediction APP")
            alcohol = st.slider("Alcohol",8,15)
            citric_acid = st.slider("Citric Acid",0,1)
            chlorides = st.slider("Chlorides",0.000,0.8)
            density = st.slider("Density",0.90,1.5)
            fixed_acidity = st.slider("Fixed Acidity",4,16)
            free_sulfur_dioxide = st.slider("Free Sulfur Dioxide",1,75)
            ph = st.slider("PH",0,14)
            residual_sugar = st.slider("Residual Sugar",0,16)
            sulphates = st.slider("Sulphates",0.00,2.00)
            total_sulfur_dioxide = st.slider("Total Sulfur Dioxide",0,300)
            volatile_acidity = st.slider("Volatile Acidity",0.00,5.00)


            res = pd.DataFrame(data = 
                {'alcohol':[alcohol],'citric_acid':[citric_acid],'chlorides':[chlorides],
                 'density':[density],'fixed_acidity':[fixed_acidity],'free_sulfur_dioxide':[free_sulfur_dioxide],
                  'ph':[ph],'residual_sugar':[residual_sugar],'sulphates':[sulphates],
                  'total_sulfur_dioxide':[total_sulfur_dioxide],
                  'volatile_acidity':[volatile_acidity]})

            
            model = st.selectbox("Model",["Desicion Tree","Random Forest"])
            if model == 'Desicion Tree':
                    dt = joblib.load('classification_app/DecisionTreeModel.pkl')
                    prediction = dt.predict(res)
                    
            prediction = str(prediction).strip('[]')
            if prediction == '0':
                prediction = "Bad"
            elif prediction == '1':
                prediction ="Good"
            else:
                prediction="Mid"
                        
            if st.button("Tahmin Et"):
                    st.write("QUALİTY :",prediction)
                    st.balloons()

    elif selected_ml_dl_app == 'Sentiment Analysis':
        with st.echo(code_location='below'):
            st.write('Sentiment Analysis App')

    elif selected_ml_dl_app == 'Object Detection':
        with st.echo(code_location='below'):
            st.write('Object Detection App')

            face_cascade = cv2.CascadeClassifier('object_detection_app/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('object_detection_app/haarcascade_eye.xml')

            def detect_faces(resim_dosya):
              new_img = np.array(resim_dosya.convert('RGB'))
              img = cv2.cvtColor(new_img,1)
              gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(gray, 1.1, 4)
              for (x, y, w, h) in faces:
                     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
              return img,faces

            def detect_eyes(resim_dosya):
              new_img = np.array(resim_dosya.convert('RGB'))
              img = cv2.cvtColor(new_img,1)
              gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
              eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
              for (ex,ey,ew,eh) in eyes:
                      cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
              return img

            resim_dosya = st.file_uploader("Resim Yukle",type=["jpg","png","jpeg"])

            if resim_dosya is not None:
                resim = Image.open(resim_dosya)
                gelistirme_turu = st.sidebar.radio("Gelistirme Turu",["Orjinal","Gri-Olcekli","Kontrast","Parlaklik","Bulaniklastirma"])

                if gelistirme_turu == 'Gri-Olcekli':
                    yeni_resim = np.array(resim.convert('RGB'))
                    img = cv2.cvtColor(yeni_resim,1)
                    gri_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    st.write("Gri Fotograf")
                    st.image(gri_img,use_column_width=True)
                elif gelistirme_turu == 'Kontrast':
                    k_oran = st.sidebar.slider("Kontrast",0.5,3.5)
                    arttirici = ImageEnhance.Contrast(resim)
                    img_cikis = arttirici.enhance(k_oran)
                    st.write("Kontrast Fotograf")
                    st.image(img_cikis,use_column_width=True)
                elif gelistirme_turu == 'Parlaklik':
                    k_oran = st.sidebar.slider("Parlaklik",0.5,3.5)
                    arttirici = ImageEnhance.Brightness(resim)
                    img_cikis = arttirici.enhance(k_oran)
                    st.write("Parlaklik Fotograf")
                    st.image(img_cikis,use_column_width=True)
                elif gelistirme_turu == 'Bulaniklastirma':
                    yeni_resim = np.array(resim.convert('RGB'))
                    B_oran = st.sidebar.slider("Bulaniklik",0.5,3.5)
                    img = cv2.cvtColor(yeni_resim,1)
                    bulanik_img = cv2.GaussianBlur(img,(11,11),B_oran)
                    st.write("Bulaniklastirma Fotograf")
                    st.image(bulanik_img,use_column_width=True)
                else:
                    st.write("Orjinal Fotograf")
                    st.image(resim,use_column_width=True)

                gorev = ["Yuz","Goz"]
                ozellik_secim = st.sidebar.selectbox("Ozellik Seciniz..",gorev)

                if st.button("Detect"):

                    if ozellik_secim == 'Yuz':
                        result_img,result_faces = detect_faces(resim)
                        st.image(result_img,use_column_width=True)
                        st.success("Found {} faces".format(len(result_faces)))

                    elif ozellik_secim == 'Goz':
                        result_img = detect_eyes(resim)
                        st.image(result_img,use_column_width=True)
                        
def sidebar():
    with st.echo(code_location='below'):
        st.sidebar.title('This is sidebar title')
        st.sidebar.header('This is sidebar header')
        st.sidebar.subheader('This is sidebar subheader')
        st.sidebar.text('This is sidebar text')
        st.sidebar.markdown(
            """
            This is sidebar markdown
            """) 
        st.sidebar.selectbox('Sidebar Selectbox',["1","2"])
        st.sidebar.slider("Sidebar Slider",1,5)
        st.sidebar.radio("Sidebar Radio",('1','2','3'))
        st.sidebar.checkbox('Sidebar Checkbox')
    
def widgets():
    with st.echo(code_location='below'):
        st.header("Widgets")

    with st.echo(code_location='below'):
        st.selectbox('Selectbox',["1","2","3"])

    with st.echo(code_location='below'):  
        st.slider("Slider",1,10)

    with st.echo(code_location='below'):
        st.radio("Radio",('1','2','3'))

    with st.echo(code_location='below'):
        st.checkbox('Checkbox','agree')

    with st.echo(code_location='below'):
        st.text_input('Text input', 'Content')

    with st.echo(code_location='below'):
        st.number_input('Number input')

    with st.echo(code_location='below'):
        st.date_input('Date input')

    with st.echo(code_location='below'):
        st.time_input('Time input')

    with st.echo(code_location='below'):
        st.text_area('Text area')

    with st.echo(code_location='below'):
        st.file_uploader('File uploader',type=['png','jpg','pdf','doc'])

    with st.echo(code_location='below'):
        st.success('Success')
        st.info('Info')
        st.warning('Warning')
        st.error('Error')

    with st.echo(code_location='below'):
        if st.button("Button"):
            st.balloons()
            
def charts():
    st.header("Charts")
    
    with st.echo(code_location='below'):
        st.subheader("Line Chart")
        line_chart_data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['a', 'b', 'c','d','e'])
        st.line_chart(line_chart_data)

    with st.echo(code_location='below'):
        st.subheader("Area Chart")
        area_chart_data = pd.DataFrame(
        np.random.randn(50, 4),
        columns=['a', 'b', 'c','d'])
        st.area_chart(area_chart_data)
    
    with st.echo(code_location='below'):
        st.subheader("Bar Chart")
        bar_chart_data = pd.DataFrame(
        np.random.randn(50, 3),
        columns=['a', 'b', 'c'])
        st.bar_chart(bar_chart_data)
    
    with st.echo(code_location='below'):
        st.subheader("Altair Chart")
        df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])
        c = alt.Chart(df).mark_circle().encode(
        x='a', y='b', size='c', color='a', tooltip=['a', 'b', 'c'])
        st.altair_chart(c, use_container_width=True)
    
    with st.echo(code_location='below'):
        st.subheader("Plotly Chart")
        x1 = np.random.randn(200) - 2
        x2 = np.random.randn(200)
        x3 = np.random.randn(200) + 2
        hist_data = [x1, x2, x3]
        group_labels = ['Group 1', 'Group 2', 'Group 3']
        fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
        st.plotly_chart(fig, use_container_width=True)
    
def maps():
    st.header("Maps")
    with st.echo(code_location='below'):
        df = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [38.491537, 27.705864],
        columns=['lat', 'lon'])
        st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=38.491537, longitude=27.705864,zoom=10,pitch=50,),
        layers=[pdk.Layer('HexagonLayer',data=df,get_position='[lon, lat]',radius=200,elevation_scale=4,
        elevation_range=[0, 100],pickable=True,extruded=True,),pdk.Layer('ScatterplotLayer',data=df,
        get_position='[lon, lat]',get_color='[200, 30, 0, 160]',get_radius=100,),],))

    with st.echo(code_location='below'):
        data1 = np.random.randn(1000, 2) / [50, 50] + [38.491537, 27.705864]
        df1 = pd.DataFrame(data1, columns=["lat", "lon"])
        viewport = {"latitude": 38.49, "longitude": 27.69, "zoom": 12, "pitch": 50}
        layers = [{"data": df1, "type": "ScatterplotLayer"}]
        st.deck_gl_chart(viewport=viewport, layers=layers)

    with st.echo(code_location='below'):
        m = folium.Map(location=[38.491537, 27.705864], zoom_start=16)
        tooltip = "HFTTF"
        folium.Marker(
            [38.491537, 27.705864], popup="HFTTF", tooltip=tooltip
        ).add_to(m)
        folium_static(m)
    
def media():
    st.header("Media")
    with st.echo(code_location='below'):
        st.subheader("İmage")
        image = Image.open('logo.png')
        st.image(image, caption='Streamlit Logo',use_column_width=True)
        st.text("FROM : https://www.streamlit.io/brand")
        st.markdown(get_binary_file_downloader_html('logo.png', 'Picture'), unsafe_allow_html=True)
    
    with st.echo(code_location='below'):
        st.subheader("Audio")
        audio_file = open('the_beatles-yesterday.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        st.text("FROM : https://www.youtube.com/watch?v=jo505ZyaCbA")
        st.markdown(get_binary_file_downloader_html('the_beatles-yesterday.mp3', 'Audio'), unsafe_allow_html=True)

    with st.echo(code_location='below'):
        st.subheader("Video")
        video_byte = open("mozart.mp4", 'rb').read()
        st.video(video_byte)
        st.text("FROM : https://www.youtube.com/watch?v=Q7UBIEoHqeM")
        st.markdown(get_binary_file_downloader_html('mozart.mp4', 'Video'), unsafe_allow_html=True)
           
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href    
    
def arayuz():
    selected_page = pages()
    if selected_page == "Main Page":
        main_page()

    elif selected_page == "ML & DL Applications":
        ml_dl_apps()

    elif selected_page == "Sidebar":
        sidebar()

    elif selected_page == "Widgets":
        widgets()

    elif selected_page == "Charts":
        charts()

    elif selected_page == "Maps":
        maps()

    elif selected_page == "Media":
        media()    

def main():
    arayuz()
    
    
if __name__ == '__main__':
    main()