import matplotlib
import streamlit 
import pandas
import joblib
import numpy
import seaborn

# id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
target = 'stroke'
categorical_labels = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
labels = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
coded_labels = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'ever_married_Yes', 'work_type_Non-working', 'work_type_Private', 'work_type_Self-employed', 'Residence_type_Urban', 'smoking_status_never smoked', 'smoking_status_smokes']

model = joblib.load('random_forest_model.pkl')
data_frame_cleand = pandas.read_csv('data_clean.csv')

if 'page' not in streamlit.session_state:
    streamlit.session_state['page'] = 'Home'

if streamlit.session_state['page'] == 'Home':
    with streamlit.form(key='form'):
        streamlit.title('Stroke Prediction')
        streamlit.header('Enter patient Information')

        gender = streamlit.selectbox('Gender', options=['Male','Female'])
        age = streamlit.number_input('Age', min_value=0, max_value=100, value=30)
        hypertension = streamlit.selectbox('Hypertension', options=['No', 'Yes'])
        heart_disease = streamlit.selectbox('Heart Disease', options=['No', 'Yes'])
        ever_married = streamlit.selectbox('Ever Married', options=['No', 'Yes'])
        work = streamlit.selectbox('Work', options=['Private', 'Govt_job', 'Never_worked','Self-employed', 'children'])
        residence = streamlit.selectbox('Residence', options=['Rural', 'Urban'])
        avg_glucose_level = streamlit.number_input('Average Glucose Level', min_value=0, max_value=300, value=180)
        bmi = streamlit.number_input('Body Mass Index', min_value=0, max_value=70, value=20)
        smoking_status = streamlit.selectbox('Smoking Status', options=['formerly smoked', 'never smoked',  'smokes'])

        if streamlit.form_submit_button('Submit'):
            streamlit.session_state['user_input'] = numpy.array([[
                gender,
                age,
                1 if hypertension == 'Yes' else 0,
                1 if heart_disease == 'Yes' else 0,
                ever_married,
                work,
                residence,
                avg_glucose_level,
                bmi,
                smoking_status,
            ]])
            streamlit.success('Input submitted! Redirecting to Result page.')
            streamlit.session_state['page'] = 'Result'
            streamlit.rerun()


elif streamlit.session_state['page'] == 'Result':
    input_data = pandas.DataFrame(streamlit.session_state['user_input'], columns=labels)
    input_data = pandas.get_dummies(input_data, columns=categorical_labels).reindex(columns=coded_labels, fill_value=0)

    streamlit.title('Stroke Prediction Results')

    prediction = model.predict(input_data)

    if prediction[0] == 1:
            streamlit.error('High Risk')
    if prediction[0] == 0:
            streamlit.success('Low Risk')

    # feature importance
    streamlit.subheader("feature importance")
    importance = model.feature_importances_
    importance_df = pandas.DataFrame({'feature': 
                                      ['age',
                                       'hypertension',
                                       'heart_disease',
                                       'avg_glucose_level',
                                       'bmi',
                                       'gender_Male',
                                       'ever_married_Yes',
                                       'work_type_Non-working',
                                       'work_type_Private',
                                       'work_type_Self-employed',
                                       'Residence_type_Urban',
                                       'smoking_status_never smoked',
                                       'smoking_status_smokes'],
                                       'importance': importance
                                       })
    importance_df = importance_df.sort_values('importance', ascending=True)

    fig4, ax4 = matplotlib.pyplot.subplots()
    ax4.barh(importance_df['feature'], importance_df['importance'])
    ax4.set_xlabel('importance Score')
    ax4.set_title('feature importance (higher = more important)')
    streamlit.pyplot(fig4)

    # glucose histogram
    streamlit.subheader("glucose level comparison")
    fig2, ax2 = matplotlib.pyplot.subplots()
    ax2.hist(data_frame_cleand[data_frame_cleand['stroke'] == 0]['avg_glucose_level'], bins=20, alpha=0.5, label='no stroke')
    ax2.hist(data_frame_cleand[data_frame_cleand['stroke'] == 1]['avg_glucose_level'], bins=20, alpha=0.5, label='stroke')
    ax2.axvline(int(float(input_data['avg_glucose_level'][0])), color='red', linestyle='dashed', linewidth=2, label='patient avg_glucose_level')
    ax2.set_xlabel('avg_glucose_level Level')
    ax2.set_ylabel('number of patients')
    ax2.legend()
    streamlit.pyplot(fig2)

    # bmi histogram
    streamlit.subheader("bmi comparison")
    fig3, ax3 = matplotlib.pyplot.subplots()
    ax3.hist(data_frame_cleand[data_frame_cleand['stroke'] == 0]['bmi'], bins=20, alpha=0.5, label='no stroke')
    ax3.hist(data_frame_cleand[data_frame_cleand['stroke'] == 1]['bmi'], bins=20, alpha=0.5, label='stroke')
    ax3.axvline(int(float(input_data['bmi'][0])), color='red', linestyle='dashed', linewidth=2, label='patient bmi')
    ax3.set_xlabel('bmi')
    ax3.set_ylabel('number of patients')
    ax3.legend()
    streamlit.pyplot(fig3)

    # age histogram
    streamlit.subheader("age comparison")
    fig5, ax4 = matplotlib.pyplot.subplots()
    ax4.hist(data_frame_cleand[data_frame_cleand['stroke'] == 0]['age'], bins=20, alpha=0.5, label='no stroke')
    ax4.hist(data_frame_cleand[data_frame_cleand['stroke'] == 1]['age'], bins=20, alpha=0.5, label='stroke')
    ax4.axvline(int(float(input_data['age'][0])), color='red', linestyle='dashed', linewidth=2, label='patient age')
    ax4.set_xlabel('age')
    ax4.set_ylabel('number of patients')
    ax4.legend()
    streamlit.pyplot(fig5)

    if streamlit.button('Return'):
        streamlit.success('Redirecting to Home page.')
        streamlit.session_state['page'] = 'Home'
        streamlit.session_state['submitted'] = False
        streamlit.rerun()