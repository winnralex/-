"""
Credit Risk ML Prototype – гибридная модель оценки кредитного риска с XAI
Единый файл: обучение CatBoost + FastAPI + SHAP интерпретация
Запуск: python app.py --train (для обучения) или python app.py --api (для запуска API)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import shap
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Конфигурация ----------
NUMERIC_COLS = ['age', 'annual_income', 'dti', 'credit_score', 'num_inquiries_6m']
CATEGORICAL_COLS = ['employment_type', 'loan_purpose']
TARGET = 'default'
RANDOM_STATE = 42

# ---------- Предобработка ----------
class Preprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.fitted = False

    def fit(self, X, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.num_imputer.fit(X[num_cols])
        self.scaler.fit(self.num_imputer.transform(X[num_cols]))
        self.cat_imputer.fit(X[cat_cols])
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Fit first")
        X_num = self.num_imputer.transform(X[self.num_cols])
        X_num_scaled = self.scaler.transform(X_num)
        X_cat = self.cat_imputer.transform(X[self.cat_cols])
        # Объединяем в DataFrame с именами колонок
        num_df = pd.DataFrame(X_num_scaled, columns=self.num_cols)
        cat_df = pd.DataFrame(X_cat, columns=self.cat_cols)
        return pd.concat([num_df, cat_df], axis=1)

    def fit_transform(self, X, cat_cols, num_cols):
        self.fit(X, cat_cols, num_cols)
        return self.transform(X)

# ---------- Модель ----------
class CreditRiskModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = None

    def train(self, df, target_col=TARGET):
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        X_processed = self.preprocessor.fit_transform(X, CATEGORICAL_COLS, NUMERIC_COLS)
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        # Оптимальные гиперпараметры (найдены заранее, можно оставить так)
        params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_seed': RANDOM_STATE,
            'verbose': 100,
            'loss_function': 'Logloss'
        }
        self.model = CatBoostClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=100)
        y_pred = self.model.predict_proba(X_val)[:, 1]
        metrics = {
            'auc': roc_auc_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred > 0.5),
            'log_loss': log_loss(y_val, y_pred)
        }
        print(f"Metrics: {metrics}")
        # Сохраняем feature names для SHAP
        self.feature_names_ = NUMERIC_COLS + CATEGORICAL_COLS
        return metrics

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)[:, 1]

    def save(self, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)

    def load(self, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

# ---------- SHAP Explainer ----------
class Explainer:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None

    def fit_background(self, X_background):
        X_proc = self.preprocessor.transform(X_background)
        self.explainer = shap.TreeExplainer(self.model)
        self.background_shap = self.explainer.shap_values(X_proc)
        return self

    def explain_local(self, X_instance):
        X_proc = self.preprocessor.transform(X_instance)
        shap_values = self.explainer.shap_values(X_proc)[0]
        expected_value = self.explainer.expected_value
        return shap_values, expected_value

    def get_reason_codes(self, X_instance, shap_values, threshold=0.05):
        feature_shap = dict(zip(self.feature_names, shap_values))
        sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        reasons = []
        for fname, shval in sorted_features[:3]:
            if abs(shval) > threshold:
                direction = "высокое" if shval > 0 else "низкое"
                reasons.append(f"Признак '{fname}' ({direction} значение) увеличивает риск")
        return reasons if reasons else ["Риск в пределах нормы"]

# ---------- FastAPI ----------
app = FastAPI(title="Credit Risk API", version="1.0")
risk_model = None
explainer = None

class PredictRequest(BaseModel):
    applicant_id: str
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    applicant_id: str
    default_probability: float
    risk_grade: str
    reason_codes: List[str]

@app.on_event("startup")
def startup():
    global risk_model, explainer
    if os.path.exists("model.joblib") and os.path.exists("preprocessor.joblib"):
        risk_model = CreditRiskModel()
        risk_model.load()
        # Для explainer нужна небольшая выборка – попробуем загрузить sample_data.csv
        if os.path.exists("sample_data.csv"):
            sample_df = pd.read_csv("sample_data.csv").head(100)
            X_sample = sample_df[NUMERIC_COLS + CATEGORICAL_COLS]
            explainer = Explainer(risk_model.model, risk_model.preprocessor, risk_model.feature_names_)
            explainer.fit_background(X_sample)
        else:
            print("No sample_data.csv, SHAP explanations limited")
    else:
        print("Model not found. Run 'python app.py --train' first")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global risk_model, explainer
    if risk_model is None or risk_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Преобразуем входные данные
    input_dict = req.features
    df = pd.DataFrame([input_dict])
    # Добавляем недостающие колонки
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = None
    proba = risk_model.predict(df)[0]
    # Risk grade
    if proba < 0.02:
        grade = "A"
    elif proba < 0.05:
        grade = "B"
    elif proba < 0.10:
        grade = "C"
    elif proba < 0.20:
        grade = "D"
    else:
        grade = "E"
    # Reason codes через SHAP
    reason_codes = []
    if explainer is not None:
        shap_vals, _ = explainer.explain_local(df)
        reason_codes = explainer.get_reason_codes(df, shap_vals)
    else:
        reason_codes = ["SHAP explanation not available"]
    return PredictResponse(
        applicant_id=req.applicant_id,
        default_probability=round(proba, 4),
        risk_grade=grade,
        reason_codes=reason_codes
    )

# ---------- Обучение при запуске с флагом --train ----------
def generate_sample_data(n=1000, seed=42):
    np.random.seed(seed)
    data = {
        'age': np.random.randint(18, 70, n),
        'annual_income': np.random.normal(60000, 30000, n).clip(10000, 200000),
        'dti': np.random.uniform(0.1, 0.6, n),
        'credit_score': np.random.normal(650, 100, n).clip(300, 850).astype(int),
        'num_inquiries_6m': np.random.poisson(2, n).clip(0, 10),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed', 'retired'], n),
        'loan_purpose': np.random.choice(['mortgage', 'car', 'debt_consolidation', 'other'], n),
        'default': np.random.binomial(1, 0.05, n)
    }
    df = pd.DataFrame(data)
    # Сделаем логическую связь: низкий кредитный рейтинг + высокий DTI -> выше вероятность дефолта
    df['default'] = ((df['credit_score'] < 600) & (df['dti'] > 0.4)).astype(int) | (df['default'] & np.random.binomial(1, 0.5, n))
    return df

def train_model():
    print("Generating sample data...")
    df = generate_sample_data(2000)
    df.to_csv("sample_data.csv", index=False)
    print("Training model...")
    model = CreditRiskModel()
    metrics = model.train(df)
    model.save()
    print(f"Model saved. Metrics: {metrics}")
    # Создаём график ROC
    from sklearn.metrics import roc_curve
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_proc = model.preprocessor.fit_transform(X, CATEGORICAL_COLS, NUMERIC_COLS)
    y_pred = model.model.predict_proba(X_proc)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    # Feature importance
    explainer_temp = Explainer(model.model, model.preprocessor, model.feature_names_)
    explainer_temp.fit_background(X.head(100))
    shap_vals = explainer_temp.explainer.shap_values(X_proc.head(100))
    plt.figure()
    shap.summary_plot(shap_vals, X_proc.head(100), feature_names=model.feature_names_, show=False)
    plt.savefig('feature_importance.png')
    print("Plots saved: roc_curve.png, feature_importance.png")

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train model and exit')
    parser.add_argument('--api', action='store_true', help='Run API server')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.api:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Usage: python app.py --train  or  python app.py --api")
