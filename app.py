import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

st.set_page_config(page_title="Prediksi & Cluster Anime", page_icon=":tada:", layout="wide")



def create_page(content, page_title=""):
    st.title(page_title)
    st.write(content)

# 1. as sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation Menu",
        options=["Prediksi Anime","Cluster Anime", "Anggota Kelompok 3 | 3 TI E"],
        icons=["book", "envelope"],
        default_index=0,
        styles={
                "nav-link-selected": {"background-color": "red"},
            },
    )

    

with st.container():
    if selected == "Prediksi Anime":
        content = """Bagian ini berfungsi untuk prediksi kepopuleran anime dengan data target type dan data latih yakni rating, members, dan episodes.
        Pada aplikasi ini kami menggunakan algoritma DecisionTreeClassifier

        """
        create_page(content, page_title=selected)
       
        uploaded_file = st.file_uploader("1. Importe le fichier csv (sous forme de tableau)", type=["csv"])
        if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df['type']=df['type'].map({'Movie':1,'TV':2,'OVA':3,'ONA':4,'Special':4})
                dataset_anime = df.dropna()
                anime2 = dataset_anime[['type','episodes','rating','members']]

                
                st.dataframe(df)

                st.write("---")
                #heatmap korelasi dataset
                dataset_corr = dataset_anime.corr()
                fig = plt.figure()
                plt.title("Correlation Heatmap", fontsize=20)
                sns.heatmap(dataset_anime.corr(), annot=True)
                st.pyplot(fig)

                st.write('Distribusi Label (type):\n', anime2['type'].value_counts())


                st.write("---")
                target_select = 'type'
                data_select = ['episodes','rating','members']

                # Pilih kolom target dan data dari dataframe
                target = anime2[target_select]
                data = anime2[data_select]
                
                #st.write(target)
                #st.write(data)
                # Tampilkan kolom target dan data yang dipilih
                st.write("Kolom target yang dipilih: ", target_select)
                
                # Pilih kolom yang hanya memiliki tipe data numerik
                col_num = df.select_dtypes(include='number').columns

                # Mengambil kolom yang memiliki tipe data numerik
                data_select = list(set(data_select) & set(col_num))

                # Pilih kolom yang tidak memiliki tipe data numerik
                data_select = st.multiselect("Pilih kolom data", data_select, default=data_select)

                

                st.write("---")
                sample_train_test_size = st.slider("pilih ukuran sampel pelatihan (rekomendasi 70%)", 0, 100, 70)
                
                xtrain, xtest, ytrain, ytest = train_test_split(data,target, train_size=sample_train_test_size/100.0, random_state=1)
                st.write("##")

                train_button = st.button("Latih model")

                
                if train_button:
                    model = DecisionTreeClassifier()
                    model.fit(xtrain, ytrain)
                    y_predict = model.predict(xtest)
                    st.write("---")
                    st.success(f'Train Accuracy: {round(model.score(xtrain, ytrain)*100,2)}%')
                    st.success(f'Predictive Data Accuracy: {round(model.score(xtest, ytest)*100,2)}%')
                    st.write(classification_report(ytest, y_predict))
                    #tampilkan heatmap
                    st.write("---")
                    fig = plt.figure()
                    plt.title("Heatmap Hasil", fontsize=20)
                    sns.heatmap(confusion_matrix(ytest, y_predict), annot=True)
                    st.pyplot(fig)

        
    if selected == "Cluster Anime":
            content = """Bagian ini berfungsi untuk melakukan kluster kepopuleran anime antara populer dengan tidak populer.
            Pada aplikasi ini kami menggunakan algoritma  KMeans
            """
            create_page(content, page_title=selected)
            st.write("---")
            uploaded_file = st.file_uploader("1. Importe le fichier csv (sous forme de tableau)", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df = df.dropna()
                df.isna().sum()
                st.dataframe(df)
                st.write("---")

                #scatter plot
                x = df.rating
                y = df.members
                fig = plt.figure()
                plt.xlabel('RATING ANIME')
                plt.ylabel('Anime Lovers')
                plt.title('Persebaran Anime Member & Anime Rating')
                plt.scatter(x,y,s=2)
                st.pyplot(fig)

                st.write("---")
                anime2 = df[['rating','members']] # mengambil kolom yang diperlukan
                anime_array = np.array(anime2) # mengubah nilai kolom rating dan member ke dalam bentuk numpy array
                #st.write(anime_array)
                #st.write("---")
                scaler = MinMaxScaler()
                xskala= scaler.fit_transform(anime_array)
                #st.write(xskala)
                #st.write("---")
                kmeans = KMeans(n_clusters=2) #menentukan berapa klaster
                kmeans.fit(xskala)
                #st.write(kmeans.cluster_centers_)
                df['Kluster'] = kmeans.labels_
                #st.write(df['Kluster'])
                warna = {0:'blue',1:'green'}
                #Scatterplot Hasil
                x = xskala[:,0]
                y = xskala[:,1]
                fig = plt.figure()
                hasil = plt.scatter(x,y,s=10,c=df['Kluster'].map(warna),marker='o',alpha=1)
                centroid = kmeans.cluster_centers_
                st.write('OUTPUT')
                plt.scatter(centroid[:,0],centroid[:,1],c='red',s=200,alpha=1)
                plt.title('Klustering K-Means')
                plt.xlabel('Rating')
                plt.ylabel('Member')
                plt.show()
                st.pyplot(fig)

                st.write('hasil dalam dataset')
                st.write('1.populer')
                anime_populer = df.loc[df['Kluster']==0]
                st.write(anime_populer)
                st.write('2. tidak populer')
                anime_unpopuler = df.loc[df['Kluster']==1]
                st.write(anime_unpopuler)




    if selected == "Anggota Kelompok 3 | 3 TI E":
        content = "1. Embun Duta Laksmana\n 2. Jessen Wind Lim\n 3. Tasya Nurul Fadila"
        create_page(content, page_title=selected)

