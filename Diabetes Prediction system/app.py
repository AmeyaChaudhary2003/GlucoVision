# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns

# Read the data
df = pd.read_csv('D:\Diabetes Prediction system\diabetes.csv')

# HEADINGS
st.markdown("<h1 style='text-align: center; color: red;'>GlucoVision</h1>", unsafe_allow_html=True)

st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(x_train, y_train)
user_result = svm.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

# Age vs Pregnancies
st.header('Pregnancy Count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
ax4 = sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
ax6 = sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)

plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
ax8 = sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color)
plt.xticks(np.arange(10,100, 5))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
ax10 = sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 900, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
ax12 = sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 3, 0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

# HEATMAP
st.subheader('Correlation Heatmap')
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = "<h2 style='text-align: center; color: red;'>Wohooo!!! You are not Diabetic</h2>"
    # st.balloons()
else:
    output = "<h2 style='text-align: center; color: red;'>You are Diabetic</h2>"

st.markdown(output, unsafe_allow_html=True)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, svm.predict(x_test)) * 100) + '%')

# Precision
st.subheader('Precision:')
precision = precision_score(y_test, svm.predict(x_test))
st.write(f"{precision * 100:.2f}%")

# Recall
st.subheader('Recall:')
recall = recall_score(y_test, svm.predict(x_test))
st.write(f"{recall * 100:.2f}%")

# F1 Score
st.subheader('F1 Score:')
f1 = f1_score(y_test, svm.predict(x_test))
st.write(f"{f1:.2f}")

# COMPARISON WITH AVERAGE VALUES
st.subheader('Comparison with Average Values')
avg_healthy = df[df['Outcome'] == 0].drop(columns=['Outcome']).mean()
avg_unhealthy = df[df['Outcome'] == 1].drop(columns=['Outcome']).mean()
user_values = user_data.iloc[0]

comparison_df = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Your Value': user_values,
    'Average Healthy Value': avg_healthy,
    'Average Unhealthy Value': avg_unhealthy
})

st.write(comparison_df)

# RISK FACTORS
st.markdown("<h3 style='color: red;'>Risk Factors</h3>", unsafe_allow_html=True)

risk_factors = {
    'Pregnancies': "Higher number of pregnancies can increase the risk of gestational diabetes.",
    'Glucose': "High glucose levels are a primary indicator of diabetes.",
    'BloodPressure': "Hypertension can be associated with diabetes.",
    'SkinThickness': "Higher skin thickness can indicate higher body fat, linked to diabetes.",
    'Insulin': "Abnormal insulin levels can indicate insulin resistance.",
    'BMI': "Higher BMI is a significant risk factor for diabetes.",
    'DiabetesPedigreeFunction': "A higher value indicates a higher genetic predisposition to diabetes.",
    'Age': "Older age increases the risk of diabetes."
}

for key, value in risk_factors.items():
    st.write(f"**{key}:** {value}")

# SUGGESTIONS AND RECOMMENDATIONS
st.markdown("<h3 style='color: green;'>Suggestions and Recommendations</h3>", unsafe_allow_html=True)
recommendations = {
    'Pregnancies': "Maintain regular check-ups during and after pregnancy.",
    'Glucose': "Monitor blood glucose levels regularly and maintain a balanced diet.",
    
    'BloodPressure': "Exercise regularly and reduce salt intake to manage blood pressure.",
    'SkinThickness': "Maintain a healthy weight through diet and exercise.",
    'Insulin': "Consult a healthcare provider for insulin management strategies.",
    'BMI': "Follow a healthy diet and exercise plan to maintain a healthy BMI.",
    'DiabetesPedigreeFunction': "Be aware of your family history and get regular screenings.",
    'Age': "Maintain a healthy lifestyle as you age to reduce the risk of diabetes."
}

for key, value in recommendations.items():
    st.write(f"**{key}:** {value}")

