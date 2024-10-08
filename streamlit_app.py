
import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
import joblib
import os
import zipfile

# Path to the ZIP file containing the models
zip_file_path = 'models.zip'
extracted_folder = 'extracted_models'

# Extract models from ZIP if not already done
def extract_models(zip_file, output_folder):
    if os.path.exists(zip_file):
        if not os.path.exists(output_folder):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
        return True
    return False

# Load models from extracted folder
def load_models(folder):
    model_1_path = os.path.join(folder, 'model_1.pkl')
    model_2_path = os.path.join(folder, 'model_2.pkl')

    if os.path.exists(model_1_path) and os.path.exists(model_2_path):
        model_1 = joblib.load(model_1_path)
        model_2 = joblib.load(model_2_path)
        return model_1, model_2
    else:
        st.error("Error: One or both model files not found.")
        return None, None
# Main Streamlit app
def main():
    st.title("ECG Heartbeat Classification")

    # Extract and load models
    if extract_models(zip_file_path, extracted_folder):
        model_1, model_2 = load_models(extracted_folder)
        if model_1 is None or model_2 is None:
            return
    else:
        st.error(f"Error: The ZIP file {zip_file_path} was not found.")
        return

    st.write("Upload a CSV file containing ECG data.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Ensure the uploaded data has 35 features (same as the model expects)
        if data.shape[1] != 35:
            st.error(f"Expected 35 features, but found {data.shape[1]}. Please upload a file with the correct number of features.")
            return

        # Preprocess the data (assuming the last column is the target)
        #x = data.drop(data.columns[-1], axis=1)

        # Scale features
        # scaler = StandardScaler()
        # x_scaled = scaler.fit_transform(data)

        # Make predictions
        predictions = model.predict(data)
        st.write("Predictions:")
        st.write(predictions)

        # Display predicted classes
        classes = ["Normal", "Abnormal Class 1", "Abnormal Class 2", "Abnormal Class 3", "Abnormal Class 4"]
        st.write("Class Labels:")
        for i in range(len(predictions)):
            st.write(f"Row {i + 1}: {classes[predictions[i]]}")

if __name__ == "__main__":
    main()
