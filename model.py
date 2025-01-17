import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

st.title('Aplikasi Pendeteksi Obat Berdasarkan Profil Kesehatan')

with st.expander('Dataset'):
    data = pd.read_csv('Classification.csv')
    st.write(data)

    st.success('Informasi Dataset')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.success('Analisa Univariat')
    deskriptif = data.describe()
    st.write(deskriptif)

# Tampilkan Tau visulisasikan
with st.expander('Visualisasi'):
    st.info('Visualisasi Per Column')
    
    fig, ax = plt.subplots()
    sns.histplot(data['Age'], color='blue')
    plt.xlabel('Age')
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.histplot(data['Na_to_K'], color='red')
    plt.xlabel('Na_to_K')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.info('Korelasi Heatmap')
    fitur_angka = ['Age', 'Na_to_K']
    matriks_korelasi = data[fitur_angka].corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(matriks_korelasi, annot=True, cmap='RdBu')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Korelasi Antar Fitur', fontsize=10)
    st.pyplot(fig)

# di sini kita plot outlier
def plot_outlier(data, column):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.boxplot(data[column])
    plt.title(f'{column} - Box Plot')
    
    plt.subplot(1,2,2)
    sns.histplot(data[column], kde=True)
    plt.title(f'{column} - Histogram')

st.pyplot(plot_outlier(data, 'Age'))
st.pyplot(plot_outlier(data, 'Na_to_K'))

# Disni kita hapus dulu outliernya
def remove_outlier(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    data = data[(data[column] >= lower) & (data[column] <= upper)]
    return data 

data = remove_outlier(data, 'Age')
data = remove_outlier(data, 'Na_to_K')

st.success('Data Setelah Outlier')
st.pyplot(plot_outlier(data, 'Age'))
st.pyplot(plot_outlier(data, 'Na_to_K'))

st.success('Data setelah outlier')
st.write(f'Dataset : {data.shape}')

# Modelkan
with st.expander('Modelling'):
    st.write('Splitting Data')
    
  
    data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})
    data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
    data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})
    
    X = data.drop(columns=['Drug'])
    y = data['Drug']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    st.success('Apply Random Forest Classifier')
    rf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred_rf)
    class_report = classification_report(y_test, y_pred_rf)
    
    st.write(f'Accuracy Score: {acc_score}')
    st.write('Classification Report:')
    st.text(class_report)

with st.sidebar:
    age = st.slider("Umur", min_value=0, max_value=100, value=50)
    sex = st.selectbox("Jenis Kelamin", options=["F", "M"])
    bp = st.selectbox("Berat Badan", options=["LOW", "NORMAL", "HIGH"])
    cholesterol = st.selectbox("Cholesterol", options=["NORMAL", "HIGH"])
    na_to_k = st.slider("Natrium Kalium Dalam Tubuh", min_value=0.0, max_value=50.0, value=20.0)

with st.expander('Hasil Prediksi'):
    sex = 0 if sex == "F" else 1
    bp = {"LOW": 0, "NORMAL": 1, "HIGH": 2}[bp]
    cholesterol = {"NORMAL": 0, "HIGH": 1}[cholesterol]
    
    data_baru = np.array([[age, sex, bp, cholesterol, na_to_k]])
    prediksi = rf_classifier.predict(data_baru).reshape(1, -1)
    st.write(prediksi)
