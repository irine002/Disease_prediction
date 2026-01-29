import joblib
import numpy as np
import pandas as pd

def predict_disease(user_symptoms):
    """
    Predicts disease based on a list of symptoms.
    
    Args:
        user_symptoms (list): List of symptom strings (e.g., ['itching', 'skin_rash'])
        
    Returns:
        str: Predicted disease
        dict: Top 3 probabilities
    """
    try:
        model = joblib.load("disease_model.pkl")
        all_symptoms = joblib.load("symptom_columns.pkl")
    except FileNotFoundError:
        return "Model not found. Please train model first.", {}

    # Create input vector
    input_vector = np.zeros(len(all_symptoms))
    
    # Match user symptoms to features
    # (Assuming user inputs match the column names exactly or roughly)
    # The dataset uses underscores, e.g. 'skin_rash'
    
    matched_symptoms = []
    for sym in user_symptoms:
        # Simple normalization: try exact match or with underscores
        sym_clean = sym.strip().lower().replace(' ', '_')
        if sym_clean in all_symptoms:
            idx = all_symptoms.index(sym_clean)
            input_vector[idx] = 1
            matched_symptoms.append(sym_clean)
        else:
            print(f"Warning: Symptom '{sym}' not recognized by model.")
            
    if not matched_symptoms:
        return "No known symptoms provided.", {}

    # Predict
    input_vector = input_vector.reshape(1, -1)
    prediction = model.predict(input_vector)[0]
    probs = model.predict_proba(input_vector)[0]
    
    # Get top 3
    top_indices = np.argsort(probs)[::-1][:3]
    top_probs = {model.classes_[i]: probs[i] for i in top_indices if probs[i] > 0}
    
    return prediction, top_probs

# Example usage
if __name__ == "__main__":
    test_symptoms = ["itching", "skin rash", "nodal skin eruptions"]
    disease, probabilities = predict_disease(test_symptoms)
    print(f"Symptoms: {test_symptoms}")
    print(f"Predicted Disease: {disease}")
    print(f"Probabilities: {probabilities}")
