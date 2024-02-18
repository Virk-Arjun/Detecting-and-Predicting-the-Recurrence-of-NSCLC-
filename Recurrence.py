import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os


from sklearn.impute import SimpleImputer


# Streamlit app
def show_predict_page():
   st.title("NSCLC Tumour Recurrence Prediction App")


   st.write("""### Please input the required information for an accurate prediction to be generated""")


   # Train the model if not trained already
   if not os.path.exists('model.joblib'):
       st.warning("The model is not trained yet. Training the model...")
       train_model()
       st.success("Model trained successfully. You can now make predictions.")


   ethnicities = [
       "Caucasian",
       "African American",
       "Asian",
       "Native Hawaiian/Pacific Islander",
       "Hispanic/Latino",
       "Other"
   ]


   # Remove the 'countries' dropdown and add parameters for prediction
   age = st.slider("Age at Histological Diagnosis", 20, 90, 40)
   gender = st.radio("Gender", ["Male", "Female"])
   smoking_status = st.radio("Smoking status", ["Never smoked", "Former smoker", "Current smoker"])
   pack_years = st.slider("Pack Years", 0, 100, 20)
   weight_lbs = st.slider("Weight (lbs)", 80, 300, 150)
   ethnicity = st.selectbox("Ethnicity", ethnicities)


   # Add Pathological T stage options
   t_stage_options = ["Not Collected", "T1a", "T1b", "T3", "T2a", "Tis", "T4"]
   pathological_t_stage = st.selectbox("Pathological T stage", t_stage_options)


   # Add Pathological M stage options
   m_stage_options = ["Not Collected", "N0", "N1", "N2"]
   pathological_m_stage = st.selectbox("Pathological M stage", m_stage_options)


   # Add Pathological N stage options
   n_stage_options = ["Not Collected", "M0", "M1a", "M1b"]
   pathological_n_stage = st.selectbox("Pathological N stage", n_stage_options)


   # Add Histopathological Grade options
   histopathological_grade_options = [
       "Not Collected",
       "G2 Moderately differentiated",
       "G1 Well differentiated",
       "Other",
       "Type I: Well to moderately differentiated",
       "G3 Poorly differentiated"
   ]
   histopathological_grade = st.selectbox("Histopathological Grade", histopathological_grade_options)


   # Add Lymphovascular invasion options
   lymphovascular_invasion_options = ["Not Collected", "Absent", "Present"]
   lymphovascular_invasion = st.selectbox("Lymphovascular invasion", lymphovascular_invasion_options)


   # Add Pleural invasion options
   pleural_invasion_options = ["Not Collected", "Yes", "No"]
   pleural_invasion = st.selectbox("Pleural invasion", pleural_invasion_options)


   # Add EGFR mutation status options
   egfr_mutation_options = ["Not Collected", "Unknown", "Wildtype", "Mutant"]
   egfr_mutation_status = st.selectbox("EGFR mutation status", egfr_mutation_options)


   # Add KRAS mutation status options
   kras_mutation_options = ["Not Collected", "Unknown", "Wildtype", "Mutant"]
   kras_mutation_status = st.selectbox("KRAS mutation status", kras_mutation_options)


   # Add ALK translocation status options
   alk_translocation_options = ["Not Collected", "Unknown", "Wildtype", "Mutant"]
   alk_translocation_status = st.selectbox("ALK translocation status", alk_translocation_options)


   # Add Adjuvant Treatment options
   adjuvant_treatment_options = ["Not Collected", "Yes", "No"]
   adjuvant_treatment = st.selectbox("Adjuvant Treatment", adjuvant_treatment_options)


   # Add Chemotherapy options
   chemotherapy_options = ["Not Collected", "Yes", "No"]
   chemotherapy = st.selectbox("Chemotherapy", chemotherapy_options)


   # Add Radiation options
   radiation_options = ["Not Collected", "Yes", "No"]
   radiation = st.selectbox("Radiation", radiation_options)


   ok = st.button("Calculate Prediction")
   # Perform prediction and display the result
   if ok:
       try:
           # Load the trained SVM model
           model = load('model.joblib')


           # Prepare input data for prediction
           input_data = {
               "Pathological T stage": [pathological_t_stage],
               "Pathological M stage": [pathological_m_stage],
               "Pathological N stage": [pathological_n_stage],
               "Histopathological Grade": [histopathological_grade],
               "Lymphovascular invasion": [lymphovascular_invasion],
               "Pleural invasion (elastic, visceral, or parietal)": [pleural_invasion],
               "EGFR mutation status": [egfr_mutation_status],
               "KRAS mutation status": [kras_mutation_status],
               "ALK translocation status": [alk_translocation_status],
               "Adjuvant Treatment": [adjuvant_treatment],
               "Chemotherapy": [chemotherapy],
               "Radiation": [radiation],
               "Age at Histological Diagnosis": [age],
               "Gender": [gender],  # Wrap the value in a list
               "Smoking status": [smoking_status],  # Wrap the value in a list
               "Pack Years": [pack_years],
               "Weight (lbs)": [weight_lbs],
               "Ethnicity_Caucasian": [1 if ethnicity == "Caucasian" else 0],
               "Ethnicity_African American": [1 if ethnicity == "African American" else 0],
               "Ethnicity_Asian": [1 if ethnicity == "Asian" else 0],
               "Ethnicity_Native Hawaiian/Pacific Islander": [
                   1 if ethnicity == "Native Hawaiian/Pacific Islander" else 0],
               "Ethnicity_Hispanic/Latino": [1 if ethnicity == "Hispanic/Latino" else 0],
               "Ethnicity_Other": [1 if ethnicity == "Other" else 0],
           }


           # Include all possible values for one-hot encoded columns
           all_possible_values = {
               "ALK translocation status_Translocated": [0],
               "ALK translocation status_Unknown": [0],
               "ALK translocation status_Wildtype": [0],
               "ALK translocation status_Not Collected": [0],
               "ALK translocation status_nan": [0],
               "Adjuvant Treatment_Yes": [0],
               "Adjuvant Treatment_No": [0],
               "Adjuvant Treatment_Not Collected": [0],
               "Adjuvant Treatment_nan": [0],
               "Chemotherapy_Yes": [0],
               "Chemotherapy_No": [0],
               "Chemotherapy_Not Collected": [0],
               "Chemotherapy_nan": [0],
               "EGFR mutation status_Not collected": [0],
               "EGFR mutation status_Unknown": [0],
               "EGFR mutation status_Wildtype": [0],
               "EGFR mutation status_Mutant": [0],
               "EGFR mutation status_nan": [0],
               "ALK translocation status_Not Collected": [0],
               "Adjuvant Treatment_No": [0],
               "Adjuvant Treatment_Not Collected": [0],
               "Adjuvant Treatment_nan": [0],
               "Ethnicity_nan": [0],
               "Gender_Male": [0],
               "Gender_nan": [0],
               "Histopathological Grade_G2 Moderately differentiated": [0],
               "Histopathological Grade_G3 Poorly differentiated": [0],
           }


           # Update the input_data dictionary with all possible values
           input_data.update(all_possible_values)


           # Convert input data to DataFrame
           input_df = pd.DataFrame(input_data)
          
            # Perform label encoding for selected parameters
              label_encoder = LabelEncoder()
              for col in parameters_to_encode:
                  if data[col].dtype == 'object':  
                      data[col] = label_encoder.fit_transform(data[col])
          
              # Save the labeled dataset to a new CSV file 
              data.to_csv('labeled_dataset.csv', index=False)

           # Make prediction using the loaded SVM model
           prediction = model.predict_proba(input_df)[:, 1]


           # Display the prediction result
           st.subheader("Prediction Result:")
           st.write(f"The predicted probability of recurrence is: {prediction[0]:.2%}")


       except Exception as e:
           st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
   show_predict_page()
