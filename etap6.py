#!/usr/bin/env python3
"""
Credit Risk ML – гибридная модель оценки кредитного риска с XAI.
Единый файл: предобработка, CatBoost, SHAP, FastAPI, обучение.
"""

import argparse
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, roc_curve
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
import joblib

# ---------- Конфигурация ----------
NUMERIC_COLS = ['age', 'annual_income', 'dti', 'credit_score', 'num_inquiries_6m']
CATEGORICAL_COLS = ['employment_type', 'loan_purpose']
TARGET = 'default'
RANDOM_STATE = 42

# ---------- Предобработка ----------
class CreditPreprocessor:
    """Предобработка данных: заполнение пропусков, нормализация, кодирование категорий."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self._num_imputer = SimpleImputer(strategy='median')
        self._scaler = StandardScaler()
        self._cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> 'CreditPreprocessor':
        """Обучить препроцессор на данных."""
        self._num_imputer.fit(X[self.numeric_cols])
        num_transformed = self._num_imputer.transform(X[self.numeric_cols])
        self._scaler.fit(num_transformed)
        self._cat_imputer.fit(X[self.categorical_cols])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Преобразовать данные."""
        if not self._fitted:
            raise RuntimeError("Препроцессор не обучен. Вызовите fit().")
        num_data = self._num_imputer.transform(X[self.numeric_cols])
        num_scaled = self._scaler.transform(num_data)
        cat_data = self._cat_imputer.transform(X[self.categorical_cols])
        result = np.hstack([num_scaled, cat_data])
        return pd.DataFrame(result, columns=self.numeric_cols + self.categorical_cols)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Обучить и преобразовать."""
        return self.fit(X).transform(X)

    @property
    def feature_names_(self) -> List[str]:
        return self.numeric_cols + self.categorical_cols


# ---------- Модель ----------
class CreditRiskModel:
    """Модель CatBoost для оценки вероятности дефолта."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        self.preprocessor = CreditPreprocessor(numeric_cols, categorical_cols)
        self.model: Optional[CatBoostClassifier] = None

    def train(self, df: pd.DataFrame, target_col: str = TARGET) -> Dict[str, float]:
        """Обучить модель на DataFrame."""
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        X_processed = self.preprocessor.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

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
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Вернуть вероятности дефолта."""
        if self.model is None:
            raise RuntimeError("Модель не обучена.")
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)[:, 1]

    def save(self, model_path: str = 'model.joblib', preproc_path: str = 'preprocessor.joblib') -> None:
        """Сохранить модель и препроцессор."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preproc_path)

    def load(self, model_path: str = 'model.joblib', preproc_path: str = 'preprocessor.joblib') -> None:
        """Загрузить модель и препроцессор."""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preproc_path)


# ---------- SHAP Explainer ----------
class ModelExplainer:
    """Интерпретация предсказаний с помощью SHAP."""

    def __init__(self, model: CatBoostClassifier, preprocessor: CreditPreprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self._explainer: Optional[shap.TreeExplainer] = None

    def fit_background(self, X_background: pd.DataFrame) -> None:
        """Инициализировать explainer на фоновых данных."""
        X_proc = self.preprocessor.transform(X_background)
        self._explainer = shap.TreeExplainer(self.model)
        # прогрев (вычисление baseline)
        self._explainer.shap_values(X_proc[:1])

    def explain_local(self, X_instance: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Вернуть SHAP-значения и expected_value для одного экземпляра."""
        if self._explainer is None:
            raise RuntimeError("Сначала вызовите fit_background().")
        X_proc = self.preprocessor.transform(X_instance)
        shap_values = self._explainer.shap_values(X_proc)[0]
        expected_value = self._explainer.expected_value
        return shap_values, expected_value

    def get_reason_codes(self, X_instance: pd.DataFrame, shap_values: np.ndarray,
                         feature_names: List[str], threshold: float = 0.05) -> List[str]:
        """Сформировать понятные причины предсказания."""
        feat_shap = dict(zip(feature_names, shap_values))
        sorted_feat = sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        reasons = []
        for fname, shval in sorted_feat[:3]:
            if abs(shval) > threshold:
                direction = "высокое" if shval > 0 else "низкое"
                reasons.append(f"Признак '{fname}' ({direction} значение) увеличивает риск")
        return reasons if reasons else ["Риск в пределах нормы"]


# ---------- Генерация синтетических данных ----------
def generate_sample_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Создать синтетический набор данных для демонстрации."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'annual_income': np.random.normal(60000, 30000, n).clip(10000, 200000),
        'dti': np.random.uniform(0.1, 0.6, n),
        'credit_score': np.random.normal(650, 100, n).clip(300, 850).astype(int),
        'num_inquiries_6m': np.random.poisson(2, n).clip(0, 10),
        'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed', 'retired'], n),
        'loan_purpose': np.random.choice(['mortgage', 'car', 'debt_consolidation', 'other'], n),
        'default': np.random.binomial(1, 0.05, n)
    })
    # Внесём логическую зависимость
    df['default'] = ((df['credit_score'] < 600) & (df['dti'] > 0.4)).astype(int) | \
                    (df['default'] & np.random.binomial(1, 0.5, n))
    return df


# ---------- FastAPI приложение ----------
app = FastAPI(title="Credit Risk API", description="Гибридная модель с SHAP", version="1.0")
_model: Optional[CreditRiskModel] = None
_explainer: Optional[ModelExplainer] = None


class PredictRequest(BaseModel):
    applicant_id: str = Field(..., description="Идентификатор заявки")
    features: Dict[str, Any] = Field(..., description="Значения признаков")


class PredictResponse(BaseModel):
    applicant_id: str
    default_probability: float
    risk_grade: str
    reason_codes: List[str]


def _get_risk_grade(prob: float) -> str:
    if prob < 0.02:
        return "A"
    if prob < 0.05:
        return "B"
    if prob < 0.10:
        return "C"
    if prob < 0.20:
        return "D"
    return "E"


@app.on_event("startup")
def startup_event() -> None:
    """Загрузить модель при старте API."""
    global _model, _explainer
    if os.path.exists("model.joblib") and os.path.exists("preprocessor.joblib"):
        _model = CreditRiskModel(NUMERIC_COLS, CATEGORICAL_COLS)
        _model.load()
        # Подготовим explainer с фоновыми данными (если есть sample_data.csv)
        if os.path.exists("sample_data.csv"):
            sample_df = pd.read_csv("sample_data.csv").head(100)
            X_sample = sample_df[NUMERIC_COLS + CATEGORICAL_COLS]
            _explainer = ModelExplainer(_model.model, _model.preprocessor)
            _explainer.fit_background(X_sample)
    else:
        print("Модель не найдена. Запустите python app.py --train")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    input_df = pd.DataFrame([req.features])
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        if col not in input_df.columns:
            input_df[col] = None

    proba = float(_model.predict(input_df)[0])
    grade = _get_risk_grade(proba)

    reason_codes = []
    if _explainer is not None:
        shap_vals, _ = _explainer.explain_local(input_df)
        reason_codes = _explainer.get_reason_codes(input_df, shap_vals, _model.preprocessor.feature_names_)
    else:
        reason_codes = ["SHAP недоступен (обучите модель с sample_data.csv)"]

    return PredictResponse(
        applicant_id=req.applicant_id,
        default_probability=round(proba, 4),
        risk_grade=grade,
        reason_codes=reason_codes
    )


# ---------- Скрипт обучения ----------
def train_and_save() -> None:
    """Обучение модели, сохранение артефактов и графиков."""
    print("Генерация синтетических данных...")
    df = generate_sample_data(2000)
    df.to_csv("sample_data.csv", index=False)

    print("Обучение модели...")
    model = CreditRiskModel(NUMERIC_COLS, CATEGORICAL_COLS)
    metrics = model.train(df)
    print(f"Метрики: {metrics}")
    model.save()

    # ROC-кривая
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_proc = model.preprocessor.transform(X)
    y_pred = model.model.predict_proba(X_proc)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    print("ROC-кривая сохранена в roc_curve.png")

    # SHAP глобальная важность
    explainer = ModelExplainer(model.model, model.preprocessor)
    explainer.fit_background(X.head(100))
    X_sample = X.head(200)
    X_sample_proc = model.preprocessor.transform(X_sample)
    shap_values = explainer._explainer.shap_values(X_sample_proc)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample_proc, feature_names=model.preprocessor.feature_names_, show=False)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("График важности признаков сохранён в feature_importance.png")


# ---------- Точка входа ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Обучить модель и сохранить")
    parser.add_argument("--api", action="store_true", help="Запустить API сервер")
    args = parser.parse_args()

    if args.train:
        train_and_save()
    elif args.api:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Использование: python app.py --train  или  python app.py --api")
