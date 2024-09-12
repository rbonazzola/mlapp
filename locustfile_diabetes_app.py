from locust import HttpUser, task

class PredictUser(HttpUser):

    @task
    def predict(self):
        
        request_body = {
            "gender": "Male",
            "age": 0,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "current",
            "bmi": 0,
            "HbA1c_level": 0,
            "blood_glucose_level": 0
        }
        
        self.client.post("/predict", json=request_body)
