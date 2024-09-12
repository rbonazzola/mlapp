import json, time, pickle, os

os.environ["KERAS_BACKEND"] = "tensorflow"


import redis
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from helpers import predict

db = redis.Redis(host="redis", port=6379, db=0)

# Load sklearn model
with open("assets/model.pkl", "rb") as f:
    model: MLPClassifier = pickle.load(f)
# Load preprocessor
with open("assets/preprocessor.pkl", "rb") as f:
    preprocessor: ColumnTransformer = pickle.load(f)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        # Pop a job from the Redis queue
        q = db.brpop("service_queue")[1]  # type: ignore

        # Decode the JSON data for the given job
        q = json.loads(q.decode("utf-8"))

        # Important! Get and keep the original job ID
        job_id = q["id"]

        # Run the loaded ml model (use the predict() function)
        prediction = predict(model, preprocessor,
            q["gender"],
            q["age"],
            q["hypertension"],
            q["heart_disease"],
            q["smoking_history"],
            q["bmi"],
            q["HbA1c_level"],
            q["blood_glucose_level"],
        )

        # Store the job results on Redis using the original job ID as the key
        db.set(job_id, json.dumps({"prediction": prediction}))

        # Sleep for a bit
        time.sleep(0.05)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
