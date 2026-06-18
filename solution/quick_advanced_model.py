import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
train_df = pd.read_csv('train_apps.csv')
test_df = pd.read_csv('test_apps.csv')

print(f"Размер train данных: {train_df.shape}")
print(f"Размер test данных: {test_df.shape}")

# Базовый feature engineering
print("\nБазовый feature engineering...")

def create_basic_features(df):
    df = df.copy()
    
    # 1. Соотношение ставок
    if 'offered_rate' in df.columns and 'cb_rate' in df.columns:
        df['rate_ratio'] = df['offered_rate'] / (df['cb_rate'].abs() + 1e-10)
        df['rate_diff'] = df['offered_rate'] - df['cb_rate']
    
    # 2. Соотношение лимитов
    if 'overdraft_limit_min' in df.columns and 'overdraft_limit_max' in df.columns:
        df['limit_range'] = df['overdraft_limit_max'] - df['overdraft_limit_min']
        
        if 'loan_amount_last' in df.columns:
            df['loan_to_limit_avg'] = df['loan_amount_last'] / ((df['overdraft_limit_min'].abs() + df['overdraft_limit_max'].abs()) / 2 + 1e-10)
    
    # 3. Признаки финансовой активности
    debt_cols = ['sum_deb_ul_90', 'sum_deb_ul_30', 'sum_deb_investment_90']
    for col in debt_cols:
        if col in df.columns:
            df[f'{col}_flag'] = (df[col].notna()).astype(int)
    
    # 4. Взаимодействия важных признаков
    if 'loan_amount_last' in df.columns and 'cnt_deb_loan_90' in df.columns:
        df['loan_debt_interaction'] = df['loan_amount_last'] * df['cnt_deb_loan_90']
    
    return df

train_df = create_basic_features(train_df)
test_df = create_basic_features(test_df)

# Подготовка данных
print("Подготовка данных...")
X = train_df.drop('target_value', axis=1)
y = train_df['target_value']

# Удаляем идентификаторы
X = X.drop(['front_id', 'decision_day'], axis=1, errors='ignore')
test_features = test_df.drop(['front_id', 'decision_day'], axis=1, errors='ignore')
test_front_ids = test_df['front_id']

# Кодирование категориальных признаков
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        combined = pd.concat([X[col], test_features[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        test_features[col] = le.transform(test_features[col].astype(str))

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_features)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмеры данных:")
print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
print(f"Test: {test_scaled.shape}")

# Обучение модели
print("\nОбучение Gradient Boosting модели...")
model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    min_samples_split=30,
    min_samples_leaf=10
)

model.fit(X_train, y_train)

# Оценка модели
y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC на validation: {roc_auc:.4f}")

# Кросс-валидация
print("Кросс-валидация...")
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Средний ROC-AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Обучение на всех данных
print("\nОбучение финальной модели на всех данных...")
final_model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    min_samples_split=30,
    min_samples_leaf=10
)

final_model.fit(X_scaled, y)

# Предсказания
print("Генерация предсказаний...")
test_predictions = final_model.predict_proba(test_scaled)[:, 1]

# Создание submission файла
submission_df = pd.DataFrame({
    'front_id': test_front_ids,
    'target_value': test_predictions
})

submission_df.to_csv('quick_advanced_submission.csv', index=False)
print(f"\nSubmission файл сохранен как 'quick_advanced_submission.csv'")

# Статистика
print(f"\nСтатистика предсказаний:")
print(f"  Средняя вероятность: {test_predictions.mean():.6f}")
print(f"  Медианная вероятность: {np.median(test_predictions):.6f}")
print(f"  Минимум: {test_predictions.min():.6f}")
print(f"  Максимум: {test_predictions.max():.6f}")

print(f"\nРаспределение по порогам:")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    count = (test_predictions >= threshold).sum()
    percentage = count / len(test_predictions) * 100
    print(f"  ≥{threshold}: {count} ({percentage:.1f}%)")

print("\nГотово! Файл для отправки: quick_advanced_submission.csv")