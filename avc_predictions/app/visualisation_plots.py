import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score


# Function to generate silhouette plot
def generate_silhouette_plot(silhouette):
    """Generate a plot for the silhouette score of clustering."""
    plt.figure(figsize=(6, 4))
    plt.barh([0], silhouette, color='skyblue')
    plt.xlim(0, 1)
    plt.xlabel('Silhouette Score')
    plt.title('Silhouette Score Plot')

    return save_plot_to_base64()


# Function to generate cluster plot
def generate_cluster_plot(X_test, model):
    """Generate an enhanced 2D cluster plot for the test data with cluster assignments and centroids."""
    cluster_labels = model.predict(X_test)

    plt.figure(figsize=(10, 8))
    # Use PCA for dimensionality reduction if the number of features is greater than 2
    if X_test.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X_test)
        x, y = reduced_data[:, 0], reduced_data[:, 1]
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
    else:
        x, y = X_test.iloc[:, 0], X_test.iloc[:, 1]
        xlabel, ylabel = X_test.columns[0], X_test.columns[1]

    sns.scatterplot(x=x, y=y, hue=cluster_labels, palette='viridis', s=120, edgecolor='black', alpha=0.8)

    # Add centroids if available
    if hasattr(model, 'cluster_centers_'):
        centroids = model.cluster_centers_
        if X_test.shape[1] > 2:
            centroids = pca.transform(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='X', label='Centroids')

    plt.title('Enhanced Cluster Plot')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    return save_plot_to_base64()


# Function to generate feature importance plot
def generate_feature_importance_plot(model):
    """Generate a feature importance plot (only if the model has feature_importances_)."""
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        features = [f"Feature {i + 1}" for i in range(len(feature_importances))]

        feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')

        return save_plot_to_base64()
    else:
        return None


# Main function to generate visualizations for supervised and unsupervised models
def generate_visualizations(X_train, X_test, y_train, y_test, model, fpr=None, tpr=None, classification_rep=None):
    """Generate various visualizations based on the model's predictions and dataset."""
    plots = {}

    if y_test is not None:
        # Supervised model: Generate metrics and visualizations for predictions

        # Generate and store the ROC curve plot if fpr and tpr are provided
        if fpr is not None and tpr is not None:
            plots['roc curve'] = generate_roc_curve_plot(fpr, tpr)

        # Generate and store the classification report plot if provided
        if classification_rep is not None:
            plots['classification report'] = generate_classification_report_plot2(classification_rep)

        # Make predictions
        predictions = model.predict(X_test)

        # Convert probabilities to class labels if predictions are 2D
        if predictions.ndim == 2:
            predicted_classes = predictions.argmax(axis=1)
        else:
            predicted_classes = predictions

        # Calculate residuals for regression models
        if y_test.ndim == 1 and predictions.ndim == 1:  # Residuals only for regression
            residuals = y_test.astype(int) - predictions.astype(int)
            plots['residual plot'] = generate_residual_plot(predictions, residuals)
            plots['residual histogram'] = generate_histogram_of_residuals(residuals)

        # Generate and store the prediction vs actual plot
        plots['prediction vs actual'] = generate_prediction_vs_actual_plot(y_test, predicted_classes)

        # Generate and store the coefficients plot
        plots['feature coefficients'] = generate_coefficients_plot(X_train, model)

    else:
        # Unsupervised model: Generate visualizations relevant to clustering

        # Cluster plot (e.g., KMeans)
        plots['cluster plot'] = generate_cluster_plot(X_test, model)

        # Generate and store a silhouette score plot for clustering models
        if hasattr(model, 'predict') and hasattr(model, 'n_clusters_'):
            cluster_labels = model.predict(X_test)
            silhouette = silhouette_score(X_test, cluster_labels)
            plots['silhouette score'] = generate_silhouette_plot(silhouette)

        # Optional: If you want a feature importances plot for unsupervised models like RandomForest
        if hasattr(model, 'feature_importances_'):
            plots['feature importance'] = generate_feature_importance_plot(model)

    return plots


def save_plot_to_base64():
    """Save the current matplotlib plot to a base64-encoded string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return image_base64


def generate_roc_curve_plot(fpr, tpr):
    """Generate and return the ROC curve plot.

        Arguments:
        - fpr: False positive rate values (array or list)
        - tpr: True positive rate values (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label="ROC Curve")
    plt.plot([0, 1], [0, 1], color='gray', linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    return save_plot_to_base64()




# Function to generate Classification Report plot
def generate_classification_report_plot(y_test, predictions):
    report = classification_report(y_test, predictions, output_dict=True)
    classification_rep = str(report)

    plt.figure(figsize=(8, 6))
    plt.text(0.1, 1.1, f"Classification Report:\n{classification_rep}", fontsize=12)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    classification_report_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return classification_report_plot

def generate_classification_report_plot2(classification_rep):
    """Generate and return the classification report plot.

        Arguments:
        - classification_rep: The classification report (string)
        """
    plt.figure(figsize=(8, 6))
    plt.text(0.1, 1.1, f"Classification Report:\n{classification_rep}", fontsize=12)
    plt.axis('off')
    return save_plot_to_base64()


def generate_residual_plot(predictions, residuals):
    """Generate and return the residual plot.

        Arguments:
        - predictions: Predicted values (array or list)
        - residuals: Residuals (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    return save_plot_to_base64()


def generate_prediction_vs_actual_plot(y_test, predictions):
    """Generate and return the prediction vs actual plot.

        Arguments:
        - y_test: Actual target values (array or list)
        - predictions: Predicted values (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    return save_plot_to_base64()


def generate_histogram_of_residuals(residuals):
    """Generate and return the histogram of residuals plot.

        Arguments:
        - residuals: Residuals (array or list)
        """
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    return save_plot_to_base64()


def generate_coefficients_plot(X_train, model):
    try:
        """Generate and return the feature coefficients plot.
    
            Arguments:
            - X_train: Training data features (DataFrame)
            - model: Trained model (object)
            """
        plt.figure(figsize=(10, 6))
        plt.bar(X_train.columns, model.coef_, alpha=0.7)
        plt.title("Feature Coefficients")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha="right")
        return save_plot_to_base64()
    except:
        pass

def generate_target_distribution_plot(data, target_column):
    """Generate and return the target distribution plot.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[target_column], kde=True, color='blue')
    plt.title(f"Distribution of {target_column}")
    return save_plot_to_base64()


def generate_correlation_heatmap(data):
    """Generate and return the correlation heatmap plot.

        Arguments:
        - data: DataFrame containing the dataset
        """
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title("Correlation Heatmap")
    return save_plot_to_base64()


def generate_bivariate_scatter_plot(data, target_column, feature):
    """Generate and return a bivariate scatter plot for target vs feature.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        - feature: Feature column name for scatter plot (string)
        """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[feature], y=data[target_column])
    plt.title(f"{target_column} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(target_column)
    return save_plot_to_base64()


def generate_pairplot(data, numerical_features, target_column):
    """Generate and return the pairplot for selected numerical features.

        Arguments:
        - data: DataFrame containing the dataset
        - numerical_features: List of numerical features (list of strings)
        - target_column: Target variable column name (string)
        """
    plt.figure(figsize=(10, 10))
    sampled_features = numerical_features[:5] + [target_column]
    sns.pairplot(data[sampled_features])
    plt.title("Pairplot of Features")
    return save_plot_to_base64()


def generate_boxplot(data, target_column, feature):
    """Generate and return the boxplot for categorical features vs target.

        Arguments:
        - data: DataFrame containing the dataset
        - target_column: Target variable column name (string)
        - feature: Categorical feature column name (string)
        """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[feature], y=data[target_column])
    plt.title(f"{target_column} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(target_column)
    return save_plot_to_base64()


# Function to generate Confusion Matrix plot
def generate_confusion_matrix_plot(y_test, predictions, model):
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    confusion_matrix_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return confusion_matrix_plot
