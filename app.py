import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config for better aesthetics
st.set_page_config(
    page_title="AI Disease Detective",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    h1 {
        color: #0e1117;
        font-family: 'Helvetica', sans-serif;
    }
    .prediction-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box_shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .stSelectbox label {
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("disease_model.pkl")
    symptoms = joblib.load("symptom_columns.pkl")
    return model, symptoms

@st.cache_resource
def load_data():
    try:
        # These files typically lack headers or have inconsistent ones, so we assign names manually
        precautions = pd.read_csv("symptom_precaution.csv", header=None, 
                                  names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
        description = pd.read_csv("symptom_Description.csv", header=None, 
                                  names=['Disease', 'Description'])
        return precautions, description
    except FileNotFoundError:
        return None, None

def main():
    st.title("🩺 AI Disease Diagnostic Assistant")
    st.markdown("### Identify potential conditions based on your symptoms")
    
    try:
        model, all_symptoms = load_model()
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'disease_model.pkl' and 'symptom_columns.pkl' are in the directory.")
        return

    precautions_df, description_df = load_data()

    # Sidebar for extra info
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg", use_container_width=True) 
        st.info("This tool uses Machine Learning to predict diseases based on symptoms. It is for educational purposes and not a substitute for professional medical advice.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Symptoms")
        # Clean symptom names for display (replace _ with space)
        formatted_symptoms = [s.replace('_', ' ').title() for s in all_symptoms]
        
        # Map back to original column names for prediction
        symptom_map = dict(zip(formatted_symptoms, all_symptoms))
        
        selected_formatted = st.multiselect(
            "What symptoms are you experiencing?",
            options=formatted_symptoms,
            placeholder="Type to search symptoms..."
        )
        
        if st.button("Analyze Symptoms"):
            if not selected_formatted:
                st.warning("Please select at least one symptom.")
            else:
                # Prepare input vector
                input_vector = np.zeros(len(all_symptoms))
                for s in selected_formatted:
                    original_name = symptom_map[s]
                    idx = all_symptoms.index(original_name)
                    input_vector[idx] = 1
                
                # Reshape for prediction
                input_vector = input_vector.reshape(1, -1)
                
                # Predict
                prediction = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]
                
                # Get confidence score
                class_idx = np.where(model.classes_ == prediction)[0][0]
                confidence = probabilities[class_idx]

                st.markdown("---")
                st.subheader("Diagnostic Result")
                
                # Display result with some flair
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #ff4b4b;">Most Likely Diagnosis:</h2>
                    <h1 style="color: #0e1117; font-size: 3em;">{prediction}</h1>
                    <p style="font-size: 1.2em; color: gray;">Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Show Description
                if description_df is not None:
                    desc_row = description_df[description_df['Disease'] == prediction]
                    if not desc_row.empty:
                        st.markdown(f"### Description")
                        st.info(desc_row.iloc[0]['Description'])

                # Show Precautions
                if precautions_df is not None:
                    prec_row = precautions_df[precautions_df['Disease'] == prediction]
                    if not prec_row.empty:
                        st.markdown("### Recommended Precautions")
                        prec_list = []
                        for i in range(1, 5):
                            val = prec_row.iloc[0].get(f'Precaution_{i}')
                            if pd.notna(val):
                                prec_list.append(val.strip().capitalize())
                        
                        for p in prec_list:
                            st.write(f"✅ {p}")

                # Show top 3 possibilities
                sorted_indices = np.argsort(probabilities)[::-1]
                st.write("")
                st.write("### Top Probable Conditions:")
                for i in range(3):
                    idx = sorted_indices[i]
                    prob = probabilities[idx]
                    if prob > 0.01: # Show only relevant ones
                        st.write(f"- **{model.classes_[idx]}**: {prob*100:.1f}%")

    with col2:
        st.markdown("### How it works")
        st.write("""
        1. **Search** for your symptoms in the dropdown.
        2. **Select** all that apply.
        3. Click **Analyze Symptoms**.
        4. The AI model compares your pattern against thousands of cases to predict the most likely condition.
        """)
        
        st.warning("⚠️ **Note**: This AI is not a doctor. Please consult a medical professional for accurate diagnosis and medication.")

if __name__ == "__main__":
    main()
