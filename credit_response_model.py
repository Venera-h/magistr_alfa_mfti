import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
train_df = pd.read_csv('train_apps.csv')
test_df = pd.read_csv('test_apps.csv')

print(f"Размер train данных: {train_df.shape}")
print(f"Размер test данных: {test_df.shape}")

# Проверка целевой переменной
print("\nАнализ целевой переменной:")
print(f"Целевая переменная: target_value")
print(f"Распределение классов:")
print(train_df['target_value'].value_counts())
print(f"Доля положительных классов: {train_df['target_value'].mean():.4f}")

# Проверка пропущенных значений
print("\nПропущенные значения в train данных:")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\nПропущенные значения в test данных:")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])

# Анализ типов данных
print("\nТипы данных:")
print(train_df.dtypes.value_counts())

# Проверка уникальных значений в категориальных признаках
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"\nКатегориальные признаки: {list(categorical_cols)}")

for col in categorical_cols:
    if col in train_df.columns:
        print(f"\n{col}:")
        print(f"  Уникальных значений: {train_df[col].nunique()}")
        print(f"  Топ-5 значений: {train_df[col].value_counts().head()}")

# Подготовка данных для моделирования
print("\nПодготовка данных...")

# Разделение на признаки и целевую переменную
X = train_df.drop('target_value', axis=1)
y = train_df['target_value']

# Удаляем front_id и decision_day для моделирования
X = X.drop(['front_id', 'decision_day'], axis=1, errors='ignore')
test_features = test_df.drop(['front_id', 'decision_day'], axis=1, errors='ignore')

# Обработка категориальных признаков
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Категориальные признаки для кодирования: {list(categorical_cols)}")

# Кодирование категориальных признаков
label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        # Объединяем train и test для кодирования
        combined = pd.concat([X[col], test_features[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        test_features[col] = le.transform(test_features[col].astype(str))
        label_encoders[col] = le

# Обработка пропущенных значений
print("\nОбработка пропущенных значений...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_features)

# Масштабирование признаков
print("Масштабирование признаков...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

# Разделение на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмеры данных:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"Test: {test_scaled.shape}")

# Обучение моделей
print("\nОбучение моделей...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train, y_train)
    
    # Предсказания на validation
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Отчет по классификации
    print(f"  Classification Report:")
    print(classification_report(y_val, y_pred))

# Выбор лучшей модели
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\nЛучшая модель: {best_model_name}")
print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

# Кросс-валидация для лучшей модели
print(f"\nКросс-валидация для {best_model_name}...")
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Средний ROC-AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Предсказания на тестовых данных
print("\nГенерация предсказаний для тестовых данных...")
test_predictions = best_model.predict_proba(test_scaled)[:, 1]

# Создание submission файла
submission_df = pd.DataFrame({
    'front_id': test_df['front_id'],
    'target_value': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print(f"Submission файл сохранен как 'submission.csv'")
print(f"Размер submission: {submission_df.shape}")

# Анализ важности признаков (для tree-based моделей)
if hasattr(best_model, 'feature_importances_'):
    print("\nТоп-10 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance.head(10))