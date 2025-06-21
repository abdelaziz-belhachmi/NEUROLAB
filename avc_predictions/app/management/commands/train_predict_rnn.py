from django.core.management.base import BaseCommand
from app.models import Patient, MyTrainedModels
from app.ML_Models import train_RNN_2
from django.contrib.auth import get_user_model
import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

User = get_user_model()


class Command(BaseCommand):
    help = 'Train the RNN model and test prediction with example patients'

    def handle(self, *args, **kwargs):
        print("✅ Starting RNN training via command...")

        # user, _ = User.objects.get_or_create(username="user3",
        #                                      defaults={"email": "test@example.com", "password": "user3user3"})

        # Train model
        results = train_RNN_2()

        print("\n📊 Training Results:")
        for k, v in results['metric_results'].items():
            print(f"{k}: {v:.4f}")

        print("\n🧪 Now predicting on sample patients...")

        high_risk_patient = {
            'gender': 'Male',
            'age': 78.0,
            'hypertension': 1,
            'heart_disease': 1,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'residence_type': 'Urban',
            'avg_glucose_level': 300.0,
            'bmi': 45.0,
            'smoking_status': 'smokes'
        }

        low_risk_patient = {
            'gender': 'Female',
            'age': 22.0,
            'hypertension': 0,
            'heart_disease': 0,
            'ever_married': 'No',
            'work_type': 'children',
            'residence_type': 'Rural',
            'avg_glucose_level': 85.0,
            'bmi': 20.0,
            'smoking_status': 'never smoked'
        }

        def predict(patient_dict):
            print(f"Input: {patient_dict}")
            user_df = pd.DataFrame([patient_dict])

            with open('media/trained_models/label_encoders.pkl', 'rb') as f:
                label_encoders_dict = pickle.load(f)

            with open('media/trained_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            for col, le in label_encoders_dict.items():
                if col in user_df:
                    user_df[col] = le.transform(user_df[col].astype(str))

            user_scaled = scaler.transform(user_df)
            user_reshaped = user_scaled.reshape(1, 1, user_scaled.shape[1])

            trained_model_instance = MyTrainedModels.objects.latest('trained_at')
            model_path = trained_model_instance.modelFile.path
            model = load_model(model_path)

            pred_prob = model.predict(user_reshaped)[0][0]
            pred_label = pred_prob > 0.5

            return {"prediction": pred_label, "probability": pred_prob}

        for label, patient in [("HIGH RISK", high_risk_patient), ("LOW RISK", low_risk_patient)]:
            print(f"\n🔍 Testing prediction for {label} patient:")
            result = predict(patient)
            print(f"Prediction: {'Stroke' if result['prediction'] else 'No Stroke'} (prob={result['probability']:.2f})")
