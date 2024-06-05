import numpy as np
import pickle
import streamlit as st

# loading the saved model
try:
    with open('trained_model.sav', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure ML Model' is present in the directory.")
    # st.stop()

def predict_diabetes(input_data):
    """
    Predicts whether a person is diabetic based on the input features.
    
    Parameters:
    input_data (list): A list containing the features in the following order:
        - Pregnancies
        - Glucose
        - BloodPressure
        - SkinThickness
        - Insulin
        - BMI
        - DiabetesPedigreeFunction
        - Age
    
    Returns:
    str: Prediction result indicating if the person is diabetic or not.
    """

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)


    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:#output is 1
        return 'The person is diabetic'
    
    
def main():
    
    st.set_page_config(page_title="Diabetes Prediction System", page_icon=":hospital:")
    st.title("Diabetes Prediction System")
    st.markdown("### Created by: [Ranjeet Kumbhar](https://github.com/RanjeetKumbhar01)")
    st.write("Enter the following details to predict diabetes:")

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

    if st.button('Get Diabetes Test Result'):
        input_data = [Pregnancies, GlucoseLevel, bloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        try:
            diagnosis = predict_diabetes(input_data)
        except Exception as e:
            st.error(f"An error occurred at Button : {e}")
        else:
            st.success(diagnosis)
            
    


if __name__ == '__main__':
    main()
# input_data = (5,166,72,19,175,25.8,0.587,51) // person is diabetic