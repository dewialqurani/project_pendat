import streamlit as st

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="014-Datamining"
)



tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Preprocessing", "Modelling", "Implementation"])


with tab1:
    st.write("Nama : Dewi Imani Al Qur Ani")
    st.write("NIM     :200411100014")
    st.write("Kelas : Data Mining C")
    st.title('Aplikasi Klasifikasi Penyakit jantung')
    st.write("""
    Penyakit jantung adalah kondisi ketika jantung mengalami gangguan. Bentuk gangguan itu sendiri bermacam-macam, bisa berupa gangguan pada pembuluh darah jantung, katup jantung, atau otot jantung. Penyakit jantung juga dapat disebabkan oleh infeksi atau kelainan lahir.
    """)
    st.write("Jantung adalah otot yang terbagi menjadi empat ruang. Dua ruang terletak di bagian atas, yaitu atrium (serambi) kanan dan kiri. Sementara dua ruang lagi terletak di bagian bawah, yaitu ventrikel (bilik) kanan dan kiri. Di antara ruang kanan dan kiri tersebut, ada dinding otot (septum) yang mencegah darah kaya oksigen bercampur dengan darah miskin oksigen.")

    st.markdown("""
    Link Dataset
    <a href="https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci"> Klik Disini</a>
    """, unsafe_allow_html=True)


    df = pd.read_csv("heart.csv")
    st.write("Dataset Penyakit Jantung : ")
    st.write(df)
    st.write("Note Nama - Nama Kolom : ")

    st.write("""
    <ol>
    <li>Age : Umur dalam satuan Tahun</li>
    <li>Sex : Jenis Kelamin (1=Laki-laki, 0=Perempuan)</li>
    <li>Cp : chest pain type (tipe sakit dada)(0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)</li>
    <li>Trestbps : tekanan darah saat dalam kondisi istirahat dalam mm/Hg</li>
    <li>Chol : serum sholestoral (kolestrol dalam darah) dalam Mg/dl </li>
    <li>Fbs : fasting blood sugar (kadar gula dalam darah setelah berpuasa) lebih dari 120 mg/dl (1=Iya, 0=Tidak)</li>
    <li>Restecg : hasil test electrocardiographic (0 = normal, 1 = memiliki kelainan gelombang ST-T (gelombang T inversi dan/atau ST elevasi atau depresi > 0,05 mV), 2 = menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes)</li>
    <li>Thalach : rata-rata detak jantung pasien dalam satu menit</li>
    <li>Exang :  keadaan dimana pasien akan mengalami nyeri dada apabila berolah raga, 0 jika tidak nyeri, dan 1 jika menyebabkan nyeri</li>
    <li>Oldpeak : depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat</li>
    <li>Slope : slope dari puncak ST setelah berolah raga. Atribut ini memiliki 3 nilai yaitu 0 untuk downsloping, 1 untuk flat, dan 2 untuk upsloping.</li>
    <li>Ca: banyaknya pembuluh darah yang terdeteksi melalui proses pewarnaan flourosopy</li>
    <li>Thal: detak jantung pasien. Atribut ini memiliki 3 nilai yaitu 1 untuk fixed defect, 2 untuk normal dan 3 untuk reversal defect</li>
    <li>Target: hasil diagnosa penyakit jantung, 0 untuk terdiagnosa positif terkena penyakit jantung koroner, dan 1 untuk negatif terkena penyakit jantung koroner.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.")
    st.write("Data preprocessing adalah proses yang penting dilakukan guna mempermudah proses analisis data. Proses ini dapat menyeleksi data dari berbagai sumber dan menyeragamkan formatnya ke dalam satu set data.")
    
    scaler = st.radio(
    "Pilih Metode Normalisasi Data : ",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_drop_column_for_minmaxscaler=df.drop(['age','trestbps','chol','thalach','oldpeak'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive bayes") #chechbox naive bayes
    knn = st.checkbox("KNN")
    ds = st.checkbox("Decission Tree")

    if nb:
        jantung = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/Pendat_UAS/main/heart.csv')
        st.write(jantung)

        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
        from sklearn.metrics import confusion_matrix

        from sklearn.model_selection import KFold,train_test_split,cross_val_score
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split

        X=jantung.iloc[:,0:4].values
        y=jantung.iloc[:,4].values

        st.write("Jumlah Shape X dan Y adalah", X.shape)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.write("Array ", y)

        #Train and Test split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        Y_pred = gaussian.predict(X_test) 
        accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

        cm = confusion_matrix(y_test, Y_pred)
        accuracy = accuracy_score(y_test,Y_pred)
        precision =precision_score(y_test, Y_pred,average='micro')
        recall =  recall_score(y_test, Y_pred,average='micro')
        f1 = f1_score(y_test,Y_pred,average='micro')
        st.write('Confusion matrix for Jantung',cm)
        st.write('accuracy_Jantung : %.3f' %accuracy)
        st.write('precision_Jantung : %.3f' %precision)
        st.write('recall_Jantung : %.3f' %recall)
        st.write('f1-score_Jantung : %.3f' %f1)

    #Metode KNN
    if knn:
        jantung = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/Pendat_UAS/main/heart.csv')
        st.write(jantung)
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        from sklearn import neighbors, datasets
        from sklearn.inspection import *
        from sklearn.model_selection import train_test_split

        n_neighbors = 5

        # import some data to play with
        jantung = datasets.load_knn()

        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        X = jantung.data[:, :5]
        y = jantung.target

        #split dataset into train an test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        # Create color maps
        cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
        cmap_bold = ["darkorange", "c", "darkblue"]

        for weights in ["uniform", "distance"]:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X_train,y_train)

            ax = plt.subplots()
            # sk.DecisionBoundaryDisplay.from_estimator(
            #     clf,
            #     X,
            #     cmap=cmap_light,
            #     ax=ax,
            #     response_method="predict",
            #     plot_method="pcolormesh",
            #     xlabel=iris.feature_names[0],
            #     ylabel=iris.feature_names[1],
            #     shading="auto",
            # )

            # Plot also the training points
            sns.scatterplot(
                x=X[:, 0],
                y=X[:, 1],
                hue=iris.target_names[y],
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
            )
            plt.title(
                "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
            )

        plt.show()

    if ds: 
        jantung = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/Pendat_UAS/main/heart.csv')
        st.write(jantung)

        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn import tree
        from matplotlib import pyplot as plt
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        y = jantung["age"]
        X = jantung.drop(columns=["age"])
        clf = tree.DecisionTreeClassifier(criterion="gini")
        clf = clf.fit(X, y)

        #plt the figure, setting a black background
        plt.figure(figsize=(10,10))
        #create the tree plot
        a = tree.plot_tree(clf,
                        rounded = True,
                        filled = True,
                        fontsize=8)
        #show the plot
        plt.show()


with tab4:
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    X=df_new.iloc[:,0:13].values
    y=df_new.iloc[:,13].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

    age=st.number_input("umur : ")
    sex=st.selectbox(
        'Pilih Jenis Kelamin',
        ('Laki-laki','Perempuan')
    )
    if sex=='Laki-laki':
        sex=1
    elif sex=='Perempuan':
        sex=0
    cp=st.selectbox(
        'Jenis nyeri dada',
        ('Typical Angina','Atypical angina','non-anginal pain','asymptomatic')
    )
    if cp=='Typical Angina':
        cp=0
    elif cp=='Atypical angina':
        cp=1
    elif cp=='non-anginal pain':
        cp=2
    elif cp=='asymptomatic':
        cp=3
    trestbps=st.number_input('resting blood pressure / tekanan darah saat kondisi istirahat(mm/Hg)')
    chol=st.number_input('serum cholestoral / kolestrol dalam darah (Mg/dl)')
    fbs=st.selectbox(
        'fasting blood sugar / gula darah puasa',
        ('Dibawah 120', 'Diatas 120')
    )
    if fbs=='Dibawah 120':
        fbs=0
    elif fbs=='Diatas 120':
        fbs=1
    restecg=st.selectbox(
        'resting electrocardiographic results',
        ('normal','mengalami kelainan gelombang ST-T','menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes')    
    )
    if restecg=='normal':
        restecg=0
    elif restecg=='mengalami kelainan gelombang ST-T':
        restecg=1
    elif restecg=='menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes':
        restecg=2
    thalach=st.number_input('thalach (rata-rata detak jantung pasien dalam satu menit)')
    exang=st.selectbox(
        'exang/exercise induced angina',
        ('ya','tidak')
    )
    if exang=='ya':
        exang=1
    elif exang=='tidak':
        exang=0
    oldpeak=st.number_input('oldpeak/depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat')
    slope=st.selectbox(
        'slope of the peak exercise',
        ('upsloping','flat','downsloping')
    )
    if slope=='upsloping':
        slope=0
    elif slope=='flat':
        slope=1
    elif slope=='downsloping':
        slope=2
    ca=st.number_input('number of major vessels')
    thal=st.selectbox(
        'Thalassemia',
        ('normal','cacat tetap','cacat reversibel')
    )
    if thal=='normal':
        thal=0
    elif thal=='cacat tetap':
        thal=1
    elif thal=='cacat reversibel':
        thal=2

    algoritma = st.selectbox(
        'pilih algoritma klasifikasi',
        ('KNN','Naive Bayes','Random Forest','Ensemble Stacking')
    )
    prediksi=st.button("Diagnosis")
    if prediksi:
        if algoritma=='KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            filename='knn.pkl'
        elif algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='Random Forest':
            model = RandomForestClassifier(n_estimators = 100)
            filename='randomforest.pkl'
        elif algoritma=='Ensemble Stacking':
            estimators = [
                ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('knn_1', KNeighborsClassifier(n_neighbors=10))             
            ]
            model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
            filename='stacking.pkl'
        
        
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        score=metrics.accuracy_score(y_test,Y_pred)

        loaded_model = pickle.load(open(filename, 'rb'))
        if scaler == 'Tanpa Scaler':
            dataArray = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        else:
            age_proceced = (age - df['age'].min(axis=0)) / (df['age'].max(axis=0) - df['age'].min(axis=0))
            trestbps_proceced = (trestbps - df['trestbps'].min(axis=0)) / (df['trestbps'].max(axis=0) - df['trestbps'].min(axis=0))
            chol_proceced = (chol - df['chol'].min(axis=0)) / (df['chol'].max(axis=0) - df['chol'].min(axis=0))
            thalach_proceced = (thalach - df['thalach'].min(axis=0)) / (df['thalach'].max(axis=0) - df['thalach'].min(axis=0))
            oldpeak_proceced = (oldpeak - df['oldpeak'].min(axis=0)) / (df['oldpeak'].max(axis=0) - df['oldpeak'].min(axis=0))
            dataArray = [age_proceced, trestbps_proceced, chol_proceced, thalach_proceced, oldpeak_proceced, sex, cp, fbs, restecg, exang, slope, ca, thal]
        pred = loaded_model.predict([dataArray])

        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit Jantung")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Memiliki penyakit Jantung")

        st.write(f"akurasi : {score}")
