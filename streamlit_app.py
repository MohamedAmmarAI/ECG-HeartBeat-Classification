import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model
model = joblib.load('model.pkl')

def main():
    st.title("ECG Heartbeat Classification")
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
