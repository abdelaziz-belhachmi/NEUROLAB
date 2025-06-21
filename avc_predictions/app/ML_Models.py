import io
import base64
import os
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.utils.timezone import now
from .models import MyTrainedModels, Patient
from django.core.files import File
from .RNN import MedicalRNN
from .visualisation_plots import generate_visualizations, generate_confusion_matrix_plot
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import LSTM
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import pandas as pd
from app.models import Patient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

import joblib


def save_model_to_disk_and_db(model_path, model_name, user, dataset_name="Stroke"):
    model_dir = os.path.join("media", "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, model_path)

    # Save to MyTrainedModels
    with open(full_path, 'rb') as model_file:
        django_file = File(model_file)
        MyTrainedModels.objects.create(
            utilisateur=user,
            modelFile=django_file,
            modelName=model_name,
            datasetName=dataset_name,
            trained_at=now()
        )




import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def get_preprocessed_data_from_patient_model(for_rnn=False):
    # Load data
    data = pd.DataFrame(list(Patient.objects.values()))
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)

    target_column = 'stroke'
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Encoding and Imputation
    label_encoders_dict = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders_dict[col] = le
        else:
            X[col] = X[col].fillna(X[col].median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save encoders and scaler for prediction reuse
    os.makedirs('media/trained_models', exist_ok=True)
    with open('media/trained_models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders_dict, f)

    with open('media/trained_models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Optional reshape for RNN
    if for_rnn:
        X_train = np.array(X_train).astype("float32").reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = np.array(X_test).astype("float32").reshape(X_test.shape[0], 1, X_test.shape[1])
    else:
        X_train = np.array(X_train).astype("float32")
        X_test = np.array(X_test).astype("float32")

    y_train = np.array(y_train).astype("int32")
    y_test = np.array(y_test).astype("int32")

    return X_train, X_test, y_train, y_test


# def get_preprocessed_data_from_patient_model():
#     data = pd.DataFrame(list(Patient.objects.values()))
#     if 'id' in data.columns:
#         data.drop(columns=['id'], inplace=True)
#
#     target_column = 'stroke'
#     y = data[target_column]
#     X = data.drop(columns=[target_column])
#
#     label_encoders_dict = {}
#     for col in X.columns:
#         if X[col].dtype == 'object':
#             X[col] = X[col].fillna(X[col].mode()[0])
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col].astype(str))
#             label_encoders_dict[col] = le
#         else:
#             X[col] = X[col].fillna(X[col].median())
#
#     scaler = StandardScaler()
#     X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#
#     # Save encoders and scaler
#     os.makedirs('media/trained_models', exist_ok=True)
#     with open('media/trained_models/label_encoders.pkl', 'wb') as f:
#         pickle.dump(label_encoders_dict, f)
#
#     with open('media/trained_models/scaler.pkl', 'wb') as f:
#         pickle.dump(scaler, f)
#
#     from sklearn.model_selection import train_test_split
#     from imblearn.over_sampling import SMOTE
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)
#
#     return X_train, X_test, y_train, y_test
#

def train_RNN_2(datasetCostumName="Stroke", authenticatedUser=None):
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense
    from sklearn.metrics import (
        accuracy_score, precision_score, f1_score, roc_auc_score,
        precision_recall_curve, confusion_matrix, roc_curve, auc
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    print("⏳ Preprocessing data...")
    X_train, X_test, y_train, y_test = get_preprocessed_data_from_patient_model(for_rnn=True)

    # Convert and reshape
    # X_train = np.array(X_train).astype("float32").reshape(X_train.shape[0], 1, X_train.shape[1])
    # X_test = np.array(X_test).astype("float32").reshape(X_test.shape[0], 1, X_test.shape[1])
    y_train = np.array(y_train).astype("int32")
    y_test = np.array(y_test).astype("int32")

    # Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Predict
    y_pred_prob = model.predict(X_test, verbose=0)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    best_threshold = thresholds[np.argmax(f1_scores)] if not np.all(f1_scores == 0) else 0.3
    y_pred = (y_pred_prob > best_threshold).astype(int)

    print(f"🔧 Best threshold based on F1: {best_threshold:.2f}")

    # Save Model
    model_filename = "RNN_model.keras"
    model_path = os.path.join("media", "trained_models", model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    save_model_to_disk_and_db(model_filename, "RNN", authenticatedUser, datasetCostumName)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
    }

    # Visualizations
    plots = {}

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plots["loss_plot"] = save_plot_to_base64()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label="Training Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plots["accuracy_plot"] = save_plot_to_base64()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plots["confusion_matrix"] = save_plot_to_base64()

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plots["roc_curve"] = save_plot_to_base64()

    print("\n✅ Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return {
        "metric_results": metrics,
        "plots": plots
    }



def train_classification_svm_classification(datasetCostumName="Stroke", authenticatedUser=None):
    X_train, X_test, y_train, y_test = get_preprocessed_data_from_patient_model(for_rnn=False)

    model = SVC(kernel='rbf', C=1.0,probability=True)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_path = "SVM_model.pkl"
    joblib.dump(model, os.path.join("media", "trained_models", model_path))

    save_model_to_disk_and_db(model_path, "SVM", authenticatedUser, datasetCostumName)


    metric_results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted', zero_division=1),
        "recall": recall_score(y_test, predictions, average='weighted', zero_division=1),
        "f1_score": f1_score(y_test, predictions, average='weighted'),
    }

    plots = {}
    try:
        plots = generate_visualizations(X_train, X_test, y_train, y_test, model)
    except Exception as e:
        print(e)

    try:
        plots['confusion matrix'] = generate_confusion_matrix_plot(y_test, predictions, model)
    except Exception as e:
        print(e)

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": remove_none_values(plots),
    }


def train_logistic_regression(datasetCostumName="Stroke", authenticatedUser=None):
    X_train, X_test, y_train, y_test = get_preprocessed_data_from_patient_model(for_rnn=False)

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
    f1 = f1_score(y_test, predictions, average='weighted')


    model_path = "LogisticRegression_model.pkl"
    joblib.dump(model, os.path.join("media", "trained_models", model_path))

    save_model_to_disk_and_db(model_path, "Logistic Regression", authenticatedUser, datasetCostumName)

    try:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        auc_score = None

    metric_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc_score,
    }

    plots = generate_visualizations(X_train, X_test, y_train, y_test, model)
    plots['confusion matrix'] = generate_confusion_matrix_plot(y_test, predictions, model)

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": remove_none_values(plots),
    }



def train_classification_random_forest(datasetCostumName="Stroke", authenticatedUser=None):
    X_train, X_test, y_train, y_test = get_preprocessed_data_from_patient_model(for_rnn=False)

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_path = "RandomForest_model.pkl"
    joblib.dump(model, os.path.join("media", "trained_models", model_path))

    save_model_to_disk_and_db(model_path, "Random Forest", authenticatedUser, datasetCostumName)


    metric_results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted', zero_division=1),
        "recall": recall_score(y_test, predictions, average='weighted', zero_division=1),
        "f1_score": f1_score(y_test, predictions, average='weighted'),
    }

    plots = generate_visualizations(X_train, X_test, y_train, y_test, model)
    plots['confusion matrix'] = generate_confusion_matrix_plot(y_test, predictions, model)

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": remove_none_values(plots),
    }



def train_lightgbm(datasetCostumName="Stroke", authenticatedUser=None):
    X_train, X_test, y_train, y_test = get_preprocessed_data_from_patient_model(for_rnn=False)

    # Prepare LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Set LightGBM parameters (you can later customize these)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    model_path = "LightGBM_model.txt"
    model.save_model(os.path.join("media", "trained_models", model_path))

    save_model_to_disk_and_db(model_path, "LightGBM", authenticatedUser, datasetCostumName)


    # Metrics
    metric_results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "auc": roc_auc_score(y_test, y_pred_prob)
    }

    # Visualizations
    plots = generate_visualizations(X_train, X_test, y_train, y_test, model)
    try:
        plots["confusion matrix"] = generate_confusion_matrix_plot(y_test, y_pred, model)
    except Exception as e:
        pass

    return {
        "metric_results": remove_none_values(metric_results),
        "plots": remove_none_values(plots)
    }




def remove_none_values(d):
    return {k: v for k, v in d.items() if v is not None}


def encode_categorical_data(data, supervised=None):
    """
    Encodes categorical data based on whether it's supervised or unsupervised.

    Args:
    - data (tuple or pd.DataFrame): If supervised, should be a tuple (X_train, X_test, y_train, y_test).
                                      If unsupervised, should be a DataFrame (df).
    - supervised (bool or None): If True, process data for supervised learning.
                                 If False, process for unsupervised learning.
                                 If None, do nothing.

    Returns:
    - Processed data (X_train, X_test, y_train, y_test or df).
    """
    if supervised is True:
        # Handle supervised case: (X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = data

        # Encode categorical features in X_train and X_test
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = X_train[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes

        return X_train, X_test, y_train, y_test

    elif supervised is False:
        # Handle unsupervised case: DataFrame (df)
        if isinstance(data, tuple):
            raise ValueError("For unsupervised learning, the input data must be a DataFrame, not a tuple.")

        df = data

        # Encode categorical features in the entire dataframe
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        return df

    elif supervised is None:
        # Do nothing if supervised is None
        return data
    else:
        raise ValueError("Invalid value for 'supervised'. It must be True, False, or None.")


def encode_categorical_data2(data, supervised=None):
    """
    Encodes categorical data based on whether it's supervised or unsupervised.

    Args:
    - data (tuple or pd.DataFrame): If supervised, should be a tuple (X, y).
                                      If unsupervised, should be a DataFrame (df).
    - supervised (bool or None): If True, process data for supervised learning.
                                 If False, process for unsupervised learning.
                                 If None, do nothing.

    Returns:
    - Processed data (X_train, X_test, y_train, y_test or df).
    """
    if supervised is True:
        # Handle supervised case: (X, y)
        X, y = data

        # Encode categorical features in X
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    elif supervised is False:
        # Handle unsupervised case: DataFrame (df)
        if isinstance(data, tuple):
            raise ValueError("For unsupervised learning, the input data must be a DataFrame, not a tuple.")

        df = data

        # Encode categorical features in the entire dataframe
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        return df

    elif supervised is None:
        # Do nothing if supervised is None
        return data
    else:
        raise ValueError("Invalid value for 'supervised'. It must be True, False, or None.")


def save_plot_to_base64():
    """Save the current matplotlib plot to a base64-encoded string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return image_base64


def check_task_type(data, target_column):
    """
    Determines if the task is classification or regression based on the target column.

    Args:
    - data (pd.DataFrame or other): The input data containing the features and target column.
    - target_column (str): The name of the target column.

    Returns:
    - str: 'classification' or 'regression' based on the target column type.
    """
    # Attempt to convert the input data to a DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            return {'error': 'Input data could not be converted to a pandas DataFrame.', 'details': str(e)}

    # Now that data is guaranteed to be a DataFrame, proceed with the logic
    try:
        # Get the target column from the dataframe
        target = data[target_column]

        # Check if target column is numeric (for regression)
        if pd.api.types.is_numeric_dtype(target):
            # If numeric, check if it has enough unique values to consider regression
            unique_values = target.nunique()

            # If there are few unique values (like 2 or 3), it's likely classification (binary or multiclass)
            if unique_values <= 10:  # This is an arbitrary threshold, you can adjust it
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    except KeyError:
        return {'error': f"Target column '{target_column}' not found in the data."}


def clean_leaderboard(leaderboard):
    cleaned_leaderboard = []
    for entry in leaderboard:
        cleaned_entry = {}
        for key, value in entry.items():
            # Remove underscores from the key
            cleaned_key = key.replace('_', ' ')
            # Remove underscores from the value if it's a string
            if isinstance(value, str):
                cleaned_value = value.replace('_', ' ')
            else:
                cleaned_value = value
            cleaned_entry[cleaned_key] = cleaned_value
        cleaned_leaderboard.append(cleaned_entry)
    return cleaned_leaderboard
