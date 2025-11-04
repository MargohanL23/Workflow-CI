# C:\Users\indah\OneDrive\Desktop\workflow-CI\MLProject\modelling.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import shutil

# ===============================================================
# 🚀 MLflow Setup
# ===============================================================
# PENTING: Hapus/Komentari URI lokal Windows. MLflow akan default ke file:./mlruns
# di folder yang sama (MLProject), yang akan dicatat oleh Actions.
# mlflow.set_tracking_uri("file:///C:/Users/indah/OneDrive/Desktop/SMSML_MARGOHAN/mlruns")
mlflow.set_experiment("MARGOHAN_CI_EXPERIMENT") # Ganti nama agar berbeda dari yang lokal

# ===============================================================
# 📥 Load Dataset
# ===============================================================
# Pastikan jalur dataset benar relatif terhadap MLProject
DATASET_PATH = "dataset_preprocessing/dataset.csv"
print(f"📥 Memuat dataset dari: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ===============================================================
# 🧹 Data Cleaning & Encoding (Sama seperti sebelumnya, diasumsikan sudah benar)
# ===============================================================
df.fillna(0, inplace=True)
df = pd.get_dummies(df, drop_first=True)
print("✅ Kolom kategorikal telah diubah menjadi numerik.")

# ===============================================================
# ✂️ Split Data
# ===============================================================
# Penyesuaian pengecekan kolom target
target_col = 'Risk_good' if 'Risk_good' in df.columns else 'Risk'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# 🧠 List Model untuk dibandingkan (Sama seperti sebelumnya)
# ===============================================================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = {}
best_model = None
best_acc = 0.0
best_model_name = ""

# ===============================================================
# 🚀 Training & Logging
# ===============================================================
for name, model in models.items():
    print(f"\n🚀 Melatih model: {name}")
    # Perbaikan: MLflow run harus berada di dalam MLProject folder,
    # tetapi dijalankan dari root (sesuai cara Actions Anda).
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # Log ke MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        # Log model ke MLflow Tracking Server
        mlflow.sklearn.log_model(model, f"model_{name}")

        print(f"✅ Akurasi {name}: {acc:.4f}")
        # print(classification_report(y_test, y_pred)) # Opsional

        results[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

# ===============================================================
# 💾 Simpan Model Terbaik dan Artefak
# ===============================================================
# Path sementara untuk model terbaik
MODEL_OUTPUT_PATH = "model_best.pkl"
joblib.dump(best_model, MODEL_OUTPUT_PATH)
print(f"\n🏆 Model terbaik ({best_model_name}) disimpan sebagai {MODEL_OUTPUT_PATH} dengan akurasi: {best_acc:.4f}")
