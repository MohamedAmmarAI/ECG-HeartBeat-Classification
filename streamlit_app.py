
import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
import joblib
import os
import zipfile
import rarfile



# Paths to the multi-part RAR files and output folder
rar_part1 = 'model.part01.rar'
rar_part2 = 'model.part02.rar'
extracted_folder = 'extracted_models'

# Extract multi-part RAR file
def extract_rar_parts(part1, part2, output_folder):
    if os.path.exists(part1) and os.path.exists(part2):
        # Use rarfile to open the first part (it will automatically combine with part 2)
        try:
            with rarfile.RarFile(part1) as rf:
                rf.extractall(output_folder)
            return True
        except rarfile.Error as e:
            st.error(f"RAR extraction failed: {str(e)}")
            return False
    else:
        st.error(f"Error: RAR parts {part1} or {part2} not found.")
        return False

# Load models from extracted folder
def load_models(folder):
    model_path = os.path.join(folder, 'model.pkl')  # Adjust to the actual model name after extraction
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        st.error("Error: Model file not found after extraction.")
        return None

# Main Streamlit app
def main():
    st.title("ECG Heartbeat Classification")

    # Extract the model from multi-part RAR
    if extract_rar_parts(rar_part1, rar_part2, extracted_folder):
        model = load_models(extracted_folder)
        if model is None:
            return
    else:
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
