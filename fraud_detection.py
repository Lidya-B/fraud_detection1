# ****************************************************************************************************
#
# File name:   fraud_detection
# Description:
#       This program checks if sum of each row, each column, and each diagonal all add up to 15.
#
# ****************************************************************************************************

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# ****************************************************************************************************

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        exit()


# ****************************************************************************************************

def explore_data(df):
    print("Dataset Overview:")
    print(df.info())
    print(df.head())
    print("Class distribution:")
    print(df['Class'].value_counts())


# ****************************************************************************************************


def visualize_class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['Class'], hue=df['Class'], palette='coolwarm', legend=False)
    plt.title('Class Distribution')
    plt.show()


# ****************************************************************************************************

def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

    return X_train, X_test, y_train, y_test


# ****************************************************************************************************

def train_model(X_train, y_train):
    print("Training the model with class weighting...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


# ****************************************************************************************************

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("Model Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    return y_pred, y_prob


# ****************************************************************************************************

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# ****************************************************************************************************

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# ****************************************************************************************************

def plot_transaction_amounts(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Amount'], color='blue', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Transaction Amount')
    plt.title('Transaction Amounts Over Time')
    plt.show()


# ****************************************************************************************************

def main():
    file_path = 'creditcard.csv'  # Change this path if needed
    df = load_dataset(file_path)
    explore_data(df)
    visualize_class_distribution(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    clf = train_model(X_train, y_train)
    y_pred, y_prob = evaluate_model(clf, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_transaction_amounts(df)
    print("Fraud detection model trained successfully.")


# ****************************************************************************************************

if __name__ == "__main__":
    main()
