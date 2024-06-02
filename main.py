import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def prediction(input):
    '''
    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    '''
    # input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:#output is 1
        return 'The person is diabetic'
    
    
def main():
    
    st.title("Welcome To Diabetes Prediction System")
   
    #get i/p data
    Pregnancies = st.text_input('No of Pregnancies')
    GlucoseLevel= st.text_input('Glucose Level')
    bloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    #prediction starts
    diagnosis =''

    if st.button('Diabates Test Result'):
        diagnosis = prediction([Pregnancies,GlucoseLevel,bloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()