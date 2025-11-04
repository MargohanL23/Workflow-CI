import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================================================
# 🚀 MLflow Setup
# ===============================================================
mlflow.set_tracking_uri("file:///C:/Users/indah/OneDrive/Desktop/SMSML_MARGOHAN/mlruns")
mlflow.set_experiment("MARGOHAN_MODEL_ADVANCED_EXPERIMENT")

# ===============================================================
# 📥 Load Dataset
# ===============================================================
print("📥 Memuat dataset...")
df = pd.read_csv("dataset.csv")
print(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ===============================================================
# 🧹 Data Cleaning
# ===============================================================
df.fillna(0, inplace=True)

# ===============================================================
# 🔢 Encode Categorical Columns
# ===============================================================
# Ubah semua kolom non-numerik menjadi numerik
df = pd.get_dummies(df, drop_first=True)
print("✅ Kolom kategorikal telah diubah menjadi numerik.")

# ===============================================================
# ✂️ Split Data
# ===============================================================
X = df.drop(columns=['Risk_good']) if 'Risk_good' in df.columns else df.drop(columns=['Risk'])
y = df['Risk_good'] if 'Risk_good' in df.columns else df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# 🧠 List Model untuk dibandingkan
# ===============================================================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = {}
best_model = None
best_acc = 0.0

# ===============================================================
# 🚀 Training & Logging
# ===============================================================
for name, model in models.items():
    print(f"\n🚀 Melatih model: {name}")
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # log ke MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, f"model_{name}")

        print(f"✅ Akurasi {name}: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = model

# ===============================================================
# 💾 Simpan Model Terbaik
# ===============================================================
joblib.dump(best_model, "model.pkl")
print(f"\n🏆 Model terbaik disimpan sebagai model.pkl dengan akurasi: {best_acc:.4f}")

print("\n🎯 Semua model berhasil dilatih dan terekam di MLflow!")
