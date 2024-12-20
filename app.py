import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"./diabetes.csv")

# Apply custom CSS (optional)
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #ff6f61;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar input
st.sidebar.header('Enter Patient Data')

def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(output, index=[0])

user_data = calc()

# Display patient data
st.subheader('Patient Data Summary')
st.write(user_data)

# Model training
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Ensure correct column alignment
user_data = user_data[x.columns]

# Prediction
result = rf.predict(user_data)

st.subheader('Prediction Result:')
output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
st.markdown(f"<h2 style='text-align: center;'>{output}</h2>", unsafe_allow_html=True)

# Model accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Model Accuracy:')
st.write(f"{accuracy:.2f}%")
