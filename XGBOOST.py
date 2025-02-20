import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os
import logging
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pfadkonfigurationen für Trainings- und Testdaten sowie Ergebnis- und Modellverzeichnisse
output_dir = '/work/oalhmad/Abschlussarbeit/New'
train_no_pca_path = f'{output_dir}/train_no_pca.csv'
test_no_pca_path = f'{output_dir}/test_no_pca.csv'
train_pca_path = f'{output_dir}/train_pca.csv'
test_pca_path = f'{output_dir}/test_pca.csv'
results_dir = '/work/oalhmad/Abschlussarbeit/Results-xgboost'
models_dir = '/work/oalhmad/Abschlussarbeit/Models-xgboost'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_data(file_path, sample_fraction):
    """Laden von Daten mit einer bestimmten Stichprobengröße."""
    logger.info(f"Daten werden geladen von: {file_path}, Stichprobengröße: {sample_fraction*100}%")
    df = pd.read_csv(file_path)
    df_sample = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
    data = df_sample.iloc[:, :-1].values  # Features extrahieren
    labels = df_sample.iloc[:, -1].values  # Labels extrahieren
    class_counts = Counter(labels)
    logger.info(f'Daten geladen von {file_path}. Klassenverteilung: {class_counts}')
    return data, labels

def plot_class_distribution(labels, title, filename):
    """Erstellen eines Plots zur Verteilung der Klassen."""
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=np.arange(min(labels), max(labels) + 2) - 0.5, edgecolor='black')
    plt.title(f'Klassenverteilung - {title}')
    plt.xlabel('Klassen')
    plt.ylabel('Anzahl')
    plt.xticks(np.arange(min(labels), max(labels) + 1))
    plt.savefig(f'{results_dir}/{filename}')
    plt.close()

def evaluate_model(X_train_scaled, y_train, skf):
    """Bewertung des XGBoost-Modells für spezifische Hyperparameter."""
    scores = []
    f1_scores = []
    for train_index, test_index in skf.split(X_train_scaled, y_train):
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Erstellen des XGBoost-DMatrizes
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dtest = xgb.DMatrix(X_test_fold, label=y_test_fold)

        # Modellparameter mit GPU-Beschleunigung
        params = {
            'objective': 'multi:softmax',  # Multiklassenklassifizierung
            'num_class': 5,  # Anzahl der Klassen
            'eval_metric': 'mlogloss',  # Log-Loss als Bewertungsmetrik
            'eta': 0.1,  # Lernrate
            'max_depth': 6,  # Maximale Tiefe der Bäume
            'tree_method': 'gpu_hist',  # Verwendung der GPU
            'scale_pos_weight': [Counter(y_train).most_common()[-1][1] / v for v in Counter(y_train).values()]  # Klassen-Gewichtung
        }

        model = xgb.train(params, dtrain, num_boost_round=100)

        y_pred = model.predict(dtest)
        accuracy = accuracy_score(y_test_fold, y_pred)
        f1 = f1_score(y_test_fold, y_pred, average='weighted')
        scores.append(accuracy)
        f1_scores.append(f1)
        logger.info(f'Fold abgeschlossen: Accuracy={accuracy}, F1-Score={f1}')
    
    return np.mean(scores), np.mean(f1_scores), model

def preprocess_and_train_xgboost(train_path, test_path, suffix):
    """Hauptfunktion zur Datenverarbeitung und Modelltraining mit XGBoost."""
    X_train, y_train = load_data(train_path, 0.03)
    X_test, y_test = load_data(test_path, 0.02)
    
    plot_class_distribution(y_train, 'Original', f'class_distribution_original_{suffix}.png')

    # Skalieren der Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Bewertung des Modells
    accuracy, f1, best_model = evaluate_model(X_train_scaled, y_train, skf)

    logger.info(f'Bestes Modell: F1-Score: {f1}')
    dump(best_model, os.path.join(models_dir, f'xgboost_model_{suffix}.joblib'))

    # Testen des besten Modells
    dtest = xgb.DMatrix(X_test_scaled)
    y_pred = best_model.predict(dtest)

    report = classification_report(y_test, y_pred)
    logger.info(f'Klassifizierungsbericht für {suffix}:\n{report}')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'Konfusionsmatrix - {suffix}')
    plt.savefig(f'{results_dir}/confusion_matrix_{suffix}.png')
    plt.close()

# Training und Evaluierung ohne PCA
preprocess_and_train_xgboost(train_no_pca_path, test_no_pca_path, 'no_pca')

# Training und Evaluierung mit PCA
preprocess_and_train_xgboost(train_pca_path, test_pca_path, 'pca')
