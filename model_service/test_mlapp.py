from helpers import predict
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
import pickle

# Load sklearn model
with open("assets/model.pkl", "rb") as f:
    model: MLPClassifier = pickle.load(f)
# Load preprocessor
with open("assets/preprocessor.pkl", "rb") as f:
    preprocessor: ColumnTransformer = pickle.load(f) 


def test_predict_lowrisk():

    parameters = {
      "gender": "Female",
      "age": 0,
      "hypertension": 0,
      "heart_disease": 0,
      "smoking_history": "never",
      "bmi": 0,
      "HbA1c_level": 0,
      "blood_glucose_level": 0
    }

    prediction = predict(model, preprocessor, **parameters)
    assert prediction <= 0.1

def test_predict_highrisk():

    parameters = {
      "gender": "Female",
      "age": 70,
      "hypertension": 1,
      "heart_disease": 1,
      "smoking_history": "current",
      "bmi": 35,
      "HbA1c_level": 8,
      "blood_glucose_level": 200
    }

    prediction = predict(model, preprocessor, **parameters)
    if prediction >= 0.95:        
        raise ValueError(f"{prediction} should be greater than 0.95")
    assert prediction > 0.95
