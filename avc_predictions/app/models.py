import os

from django.contrib.auth.models import AbstractUser
from django.core.files import File
from django.db import models
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.core.files.storage import default_storage
import matplotlib.pyplot as plt
import seaborn as sns

# from theMLhub import settings

from django.conf import settings

class Utilisateur(AbstractUser):
    profile_picture_path = models.FileField(upload_to='Profile_pictures/', null=True, blank=True)
    country = models.CharField(max_length=100, null=True, blank=True)
    full_name = models.CharField(max_length=150, null=True, blank=True)
    STATUS_CHOICES = [
        ('Personal', 'Personal'),
        ('Organisation (hospital/clinique)', 'Organisation (hospital/clinique)'),
    ]
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, null=True, blank=True)


class RawDataset(models.Model):
    utilisateur = models.ForeignKey(Utilisateur, on_delete=models.CASCADE)
    file_raw_dataset = models.FileField(upload_to='raw_datasets/')
    datasetCostumName = models.CharField(max_length=100, default='DataSetFile' ,null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    TargetColumn = models.CharField(max_length=255, null=True,blank=True,default='target')
    VISIBILITY_CHOICES = [
        ('Public', 'Public'),
        ('Private', 'Private'),
    ]
    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES, default='Private')

    def __str__(self):
        return self.datasetCostumName
    #load the data from the file

    import os
    from django.conf import settings
    import pandas as pd
    from django.conf import settings
    import os

    from django.conf import settings
    import os

    def generate_visualizations(self):

        raw_file_path = self.file_raw_dataset.path
        if raw_file_path.endswith('.csv'):
            df = pd.read_csv(raw_file_path)
        elif raw_file_path.endswith('.xls') or raw_file_path.endswith('.xlsx'):
            df = pd.read_excel(raw_file_path, engine='openpyxl' if raw_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {raw_file_path}")

        # Directory to save visualizations (relative to MEDIA_ROOT)
        visualizations_dir = os.path.join(settings.MEDIA_ROOT, 'data_visualizations', self.datasetCostumName)
        os.makedirs(visualizations_dir, exist_ok=True)

        # Generate and save graphs
        output_paths = []
        correlation_path = os.path.join(visualizations_dir, 'correlation_heatmap.png')
        DataVisualization.generate_correlation_heatmap(df, correlation_path)
        output_paths.append(correlation_path)

        if self.TargetColumn in df.columns:
            class_distribution_path = os.path.join(visualizations_dir, 'class_distribution.png')
            # DataVisualization.generate_class_distribution(df, self.TargetColumn, class_distribution_path)
            # output_paths.append(class_distribution_path)
            try:
                DataVisualization.generate_class_distribution(df, self.TargetColumn, class_distribution_path)
                success = True
            except Exception as e:
                success = False
                print(f'exception class distribution visual : {e}')
            if success:
                output_paths.append(class_distribution_path)

        histogram_path = os.path.join(visualizations_dir, 'histograms.png')
        DataVisualization.generate_histograms(df, histogram_path)
        output_paths.append(histogram_path)

        for path in output_paths:
            
            visualization_name = os.path.basename(path).replace('.png', '').replace('_', ' ').title()
            relative_path = os.path.relpath(path, settings.MEDIA_ROOT)
            DataVisualization.objects.create(
                dataset=self,
                visualization_name=visualization_name,
                graph_type=visualization_name,
                graph_file=relative_path ,
            )

from django.http import JsonResponse

class PreprocessedDataset(models.Model):
    raw_dataset = models.ForeignKey(RawDataset, on_delete=models.CASCADE)
    file_preprocessed_data = models.FileField(upload_to='processed_datasets/')
    processed_at = models.DateTimeField(auto_now_add=True)
    preprocessedCostumName = models.CharField(max_length=100, null=True, blank=True)
    VISIBILITY_CHOICES = [
        ('Public', 'Public'),
        ('Private', 'Private'),
    ]
    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES, default='Private')

    def process_data(self, target_column):
        raw_file_path = self.raw_dataset.file_raw_dataset.path
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"The file {raw_file_path} does not exist.")
        
        if raw_file_path.endswith('.csv'):
            df = pd.read_csv(raw_file_path)
        elif raw_file_path.endswith('.xls') or raw_file_path.endswith('.xlsx'):
            df = pd.read_excel(raw_file_path, engine='openpyxl' if raw_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {raw_file_path}")

        if target_column not in df.columns:
            raise ValueError(f"The target column '{target_column}' is not found in the dataset.")
        #
        # if df[target_column].dtype == 'object':
        #     df[target_column] = df[target_column].astype('category').cat.codes
        #
        # df = df.drop_duplicates()
        #
        # for col in df.columns:
        #     if df[col].dtype == 'object':
        #         df[col] = df[col].fillna(df[col].mode()[0])
        #     else:
        #         df[col] = df[col].fillna(df[col].median())
        #

        df = df.drop(columns=["id"], errors='ignore')

        X = df.drop(target_column, axis=1)

        y = df[target_column]


        if pd.api.types.is_numeric_dtype(y):
            pass
        else:
            if y.value_counts().min() < 0.6 * y.value_counts().max():
                smote = SMOTE()
                X, y = smote.fit_resample(X, y)
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        # numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        # sc = StandardScaler()
        # X_train[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
        # X_test[numeric_cols] = sc.transform(X_test[numeric_cols])
        #
        # train_data = pd.DataFrame(X_train, columns=X.columns)
        # train_data['target'] = y_train
        # train_data['set'] = 'train'
        #
        # test_data = pd.DataFrame(X_test, columns=X.columns)
        # test_data['target'] = y_test
        # test_data['set'] = 'test'
        #
        # processed_data = pd.concat([train_data, test_data], ignore_index=True)
        #
        # processed_datasets_dir = os.path.join('media', 'processed_datasets')
        # os.makedirs(processed_datasets_dir, exist_ok=True)
        # processed_file_path = os.path.join(processed_datasets_dir, f'{self.preprocessedCostumName}.csv')
        #
        # processed_data.to_csv(processed_file_path, index=False)
        #
        # # Associate the processed file with the file_preprocessed_data field
        # with open(processed_file_path, 'rb') as f:
        #     self.file_preprocessed_data.save(f'{self.preprocessedCostumName}.csv', File(f))
        #
        # # Save the model instance
        # self.save()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

     
        return X,y


    def process_data_unsupervised(self):
        # Load the raw dataset
        raw_file_path = self.raw_dataset.file_raw_dataset.path
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"The file {raw_file_path} does not exist.")
        
        if raw_file_path.endswith('.csv'):
            df = pd.read_csv(raw_file_path)
        elif raw_file_path.endswith('.xls') or raw_file_path.endswith('.xlsx'):
            df = pd.read_excel(raw_file_path, engine='openpyxl' if raw_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {raw_file_path}")

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':  # For categorical columns
                df[col] = df[col].fillna(df[col].mode()[0])
            else:  # For numerical columns
                df[col] = df[col].fillna(df[col].median())

        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        # Normalize numerical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Save the processed dataset
        processed_datasets_dir = os.path.join('media', 'processed_datasets')
        os.makedirs(processed_datasets_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_datasets_dir, f'{self.preprocessedCostumName}+_processed.csv')

        df.to_csv(processed_file_path, index=False)

        # Associate the processed file with the file_preprocessed_data field
        with open(processed_file_path, 'rb') as f:
            self.file_preprocessed_data.save(f'{self.preprocessedCostumName}_processed.csv', File(f))

        # Save the model instance
        self.save()

        return df

    def generate_visualizations(self):
        # Vérifier si le fichier prétraité est associé à l'attribut
        if not self.file_preprocessed_data:
            raise ValueError("No file is associated with the 'file_preprocessed_data' attribute.")
        
        # Charger le fichier prétraité
        preprocessed_file_path = self.file_preprocessed_data.path
        if preprocessed_file_path.endswith('.csv'):
            df = pd.read_csv(preprocessed_file_path)
        elif preprocessed_file_path.endswith('.xls') or preprocessed_file_path.endswith('.xlsx'):
            df = pd.read_excel(preprocessed_file_path, engine='openpyxl' if preprocessed_file_path.endswith('.xlsx') else 'xlrd')
        else:
            raise ValueError(f"Unsupported file type: {preprocessed_file_path}")

        # Répertoire pour enregistrer les visualisations (relatif à MEDIA_ROOT)
        visualizations_dir = os.path.join(settings.MEDIA_ROOT, 'data_visualizations', self.preprocessedCostumName)
        os.makedirs(visualizations_dir, exist_ok=True)

        # Générer et sauvegarder les graphiques
        output_paths = []
        correlation_path = os.path.join(visualizations_dir, 'correlation_heatmap.png')
        DataVisualization.generate_correlation_heatmap(df, correlation_path)
        output_paths.append(correlation_path)

        histogram_path = os.path.join(visualizations_dir, 'histograms.png')
        DataVisualization.generate_histograms(df, histogram_path)
        output_paths.append(histogram_path)

        if 'target' in df.columns:
            class_distribution_path = os.path.join(visualizations_dir, 'class_distribution.png')
            try:
                DataVisualization.generate_class_distribution(df, 'target', class_distribution_path)
                success = True
            except Exception as e:
                success = False
                print(f'exception class distribution visual : {e}')
            if success:
                output_paths.append(class_distribution_path)

        # Enregistrer les visualisations dans la base de données
        for path in output_paths:
            visualization_name = os.path.basename(path).replace('.png', '').replace('_', ' ').title()
            relative_path = os.path.relpath(path, settings.MEDIA_ROOT)
            DataVisualization.objects.create(
                dataset=self.raw_dataset,
                dataset_processed=self,
                visualization_name=visualization_name,
                graph_type=visualization_name,
                graph_file=relative_path,
            )

class DataVisualization(models.Model):
    dataset = models.ForeignKey(RawDataset, on_delete=models.CASCADE)  
    dataset_processed = models.ForeignKey(PreprocessedDataset, on_delete=models.CASCADE, null=True, blank=True)
    visualization_name = models.CharField(max_length=100, default='Visualization')
    graph_type = models.CharField(max_length=50)  
    graph_file = models.FileField(upload_to='data_visualizations/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.visualization_name} ({self.graph_type})"
    @staticmethod
    def generate_correlation_heatmap(df, output_path):
        # Ensure only numeric columns are used for correlation heatmap
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation heatmap.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def generate_class_distribution(df, column, output_path):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        
        # Handle non-numeric columns
        if df[column].dtype == 'object':
            plt.figure(figsize=(8, 6))
            sns.countplot(x=df[column])
            plt.title(f'Class Distribution of {column}')
            plt.xlabel('Classes')
            plt.ylabel('Frequency')
            plt.savefig(output_path)
            plt.close()
        else:
            raise ValueError(f"Column '{column}' is not categorical.")

    @staticmethod
    def generate_histograms(df, output_path):
        # Only select numeric columns for histograms
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_cols.empty:
            raise ValueError("No numeric columns found for histograms.")
        
        df[numeric_cols].hist(figsize=(12, 10), bins=20)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class AiModel(models.Model):
    name = models.CharField(max_length=100)
    model_params = models.JSONField()  # For flexibility


class Result(models.Model):
    utilisateur = models.ForeignKey(Utilisateur, on_delete=models.CASCADE)
    ai_model = models.ForeignKey(AiModel, on_delete=models.CASCADE)  # Many-to-one with AiModel
    preprocessed_dataset = models.ForeignKey(PreprocessedDataset,null=True,blank=True, on_delete=models.CASCADE)  # Many-to-one with PreprocessedDataset
    resultobject = models.JSONField(default=dict,null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Track when the result was created


from django.utils import timezone


class MyTrainedModels(models.Model):
    utilisateur = models.ForeignKey(Utilisateur, null=True,blank=True, on_delete=models.CASCADE)
    modelFile = models.FileField(upload_to='trained_models/')  # This saves to MEDIA_ROOT/trained_models/
    modelName = models.CharField(max_length=100, default="Rnn")
    datasetName = models.CharField(max_length=100, default="stroke_dataset")
    trained_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.modelName} trained on {self.datasetName} at {self.trained_at}"

class Patient(models.Model):
    GENDER_CHOICES = [('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')]
    MARRIED_CHOICES = [('Yes', 'Yes'), ('No', 'No')]
    WORK_CHOICES = [
        ('Private', 'Private'),
        ('Self-employed', 'Self-employed'),
        ('Govt_job', 'Govt_job'),
        ('children', 'children'),
        ('Never_worked', 'Never_worked'),
    ]
    RESIDENCE_CHOICES = [('Urban', 'Urban'), ('Rural', 'Rural')]
    SMOKING_CHOICES = [
        ('formerly smoked', 'formerly smoked'),
        ('never smoked', 'never smoked'),
        ('smokes', 'smokes'),
        ('Unknown', 'Unknown'),
        ('N/A', 'N/A'),
    ]

    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.IntegerField()
    hypertension = models.BooleanField()
    heart_disease = models.BooleanField()
    ever_married = models.CharField(max_length=3, choices=MARRIED_CHOICES)
    work_type = models.CharField(max_length=20, choices=WORK_CHOICES)
    residence_type = models.CharField(max_length=10, choices=RESIDENCE_CHOICES)
    avg_glucose_level = models.FloatField()
    bmi = models.FloatField(null=True, blank=True)
    smoking_status = models.CharField(max_length=20, choices=SMOKING_CHOICES)
    stroke = models.BooleanField()

    def __str__(self):
        return f"{self.gender} - Age {self.age}"


class AnalyseResult(models.Model):
    utilisateur = models.ForeignKey(Utilisateur, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    prediction = models.CharField(max_length=25)  # "Stroke" or "No Stroke"
    proba = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    input_data = models.JSONField()

    def __str__(self):
        return f"{self.utilisateur.username} - {self.model_name} - {self.prediction}"