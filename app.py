import matplotlib
import sklearn
import streamlit 
import sklearn
import pandas
import joblib
import numpy
import seaborn

# id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
target = 'stroke'
numerical_labels = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical_labels = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
labels = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
coded_labels = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'ever_married_Yes', 'work_type_Non-working', 'work_type_Private', 'work_type_Self-employed', 'Residence_type_Urban', 'smoking_status_never smoked', 'smoking_status_smokes']

model = joblib.load('random_forest_model.pkl')
data_frame_cleand = pandas.read_csv('data_clean.csv')

X = data_frame_cleand.drop(columns='stroke')
y = data_frame_cleand['stroke']

X_encoded = pandas.get_dummies(X, columns=categorical_labels, drop_first=True)
X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

if 'page' not in streamlit.session_state:
    streamlit.session_state['page'] = 'Home'

if streamlit.session_state['page'] == 'Home':
    with streamlit.form(key='form'):
        streamlit.title('Stroke Prediction')
        streamlit.header('Enter patient Information')

        gender = streamlit.selectbox('Gender', options=['Male','Female'])
        age = streamlit.number_input('Age', min_value=0.0, max_value=100.0, value=67.0)
        hypertension = streamlit.selectbox('Hypertension', options=['0', '1'])
        heart_disease = streamlit.selectbox('Heart Disease', options=['1', '0'])
        ever_married = streamlit.selectbox('Ever Married', options=['Yes', 'No'])
        work = streamlit.text_input('Work', value='Private')
        residence = streamlit.selectbox('Residence', options=['Urban', 'Rural'])
        avg_glucose_level = streamlit.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=228.69)
        bmi = streamlit.number_input('Body Mass Index', min_value=0.0, max_value=70.0, value=25.0, format='%.1f')
        smoking_status = streamlit.selectbox('Smoking Status', options=['formerly smoked', 'never smoked',  'smokes'])

        if streamlit.form_submit_button('Submit'):
            streamlit.session_state['user_input'] = numpy.array([[
                gender,
                age,
                hypertension,
                heart_disease,
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
    pandas.set_option('display.width', 1000)
    pandas.set_option('display.max_colwidth', None)

    input_data = pandas.DataFrame(streamlit.session_state['user_input'], columns=labels)
    print(input_data)
    print(X.head(1))
    
    test = pandas.get_dummies(X.head(1), columns=categorical_labels)
    print(test)


    input_data = pandas.get_dummies(input_data, columns=categorical_labels)
    input_data = input_data.reindex(columns=coded_labels, fill_value=0)
    print(input_data)

    streamlit.title('Stroke Prediction Results')

    prediction = model.predict(input_data)

    streamlit.success(f'The random forest model predicts: {prediction[0]}')
    if prediction[0] == 1:
            streamlit.error('The random forest model predicts: High Risk')
    if prediction[0] == 0:
            streamlit.success('The random forest model predicts: Low Risk')

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

    # # confusion matrix
    # y_pred = model.predict(X_test)
    # matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # cm_percentage = matrix.astype('float') / matrix.sum(axis=1)[:, numpy.newaxis] * 100
    # # cm_percentage = matrix.astype('float') / matrix.sum() * 100
    # display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["no stroke", "stroke"])
    # streamlit.subheader('random tree confusion matrix')
    # # seaborn.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["no stroke", "stroke"], yticklabels=["no stroke", "stroke"])
    # figure, ax = matplotlib.pyplot.subplots(figsize=(5, 5))
    # display.plot(ax=ax, values_format=".2f")
    # streamlit.pyplot(figure)

    if streamlit.button('Return'):
        streamlit.success('Redirecting to Home page.')
        streamlit.session_state['page'] = 'Home'
        streamlit.session_state['submitted'] = False
        streamlit.rerun()