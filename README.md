# AI Disease Prediction System 🩺

A Machine Learning-powered healthcare assistant that predicts potential diseases based on user symptoms. The system uses a Random Forest Classifier to analyze symptoms and provides disease descriptions along with precautionary measures.

## 🚀 Features

-   **Symptom-Based Prediction**: Select your symptoms from a comprehensive list to get a diagnosis.
-   **Machine Learning**: Powered by a Random Forest algorithm trained on a dataset of 40+ diseases.
-   **Detailed Analysis**: Provides the predicted disease, confidence score, and top 3 alternative probabilities.
-   **Helper Information**: Displays a brief description of the disease and recommended precautions.
-   **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/) for a clean and interactive web experience.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/irine002/Disease_prediction.git
    cd Disease_prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas scikit-learn streamlit
    ```

3.  **Download Data & Train Model** (First Run):
    The project includes the necessary scripts to handle data.
    ```bash
    # Download dataset
    python download_data.py
    
    # Train the model
    python train_model.py
    ```

##  ▶️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
```
This will open the app in your default web browser (usually at `http://localhost:8501`).

## 📂 Project Structure

-   `app.py`: Main Streamlit application file.
-   `train_model.py`: Script to train the Machine Learning model.
-   `download_data.py`: Script to download the necessary datasets.
-   `predictor.py`: Standalone script for command-line predictions.
-   `dataset.csv`: The training dataset.
-   `symptom_Description.csv` & `symptom_precaution.csv`: Auxiliary medical data.

## ⚠️ Disclaimer

**This tool is for educational purposes only.** It uses AI to predict diseases based on general symptoms and should **NOT** be considered a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
