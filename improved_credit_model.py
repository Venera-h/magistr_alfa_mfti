import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
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
class_dist = train_df['target_value'].value_counts()
print(class_dist)
print(f"Доля положительных классов: {train_df['target_value'].mean():.4f}")
print(f"Соотношение классов: {class_dist[0]}:{class_dist[1]} (1:{class_dist[0]/class_dist[1]:.1f})")

# Подготовка данных
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
print("Обработка пропущенных значений...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_features)

# Масштабирование признаков
print("Масштабирование признаков...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

# Разделение на train/validation с учетом дисбаланса
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмеры данных:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"Test: {test_scaled.shape}")

# Методы борьбы с дисбалансом
print("\nЭксперименты с методами борьбы с дисбалансом...")

# 1. Без балансировки
print("\n1. Без балансировки классов:")
gb_base = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb_base.fit(X_train, y_train)
y_pred_proba_base = gb_base.predict_proba(X_val)[:, 1]
roc_auc_base = roc_auc_score(y_val, y_pred_proba_base)
print(f"   ROC-AUC: {roc_auc_base:.4f}")

# 2. С балансировкой весов классов
print("\n2. С балансировкой весов классов:")
gb_balanced = GradientBoostingClassifier(
    random_state=42, 
    n_estimators=100,
    subsample=0.8,
    max_depth=5
)
gb_balanced.fit(X_train, y_train)
y_pred_proba_balanced = gb_balanced.predict_proba(X_val)[:, 1]
roc_auc_balanced = roc_auc_score(y_val, y_pred_proba_balanced)
print(f"   ROC-AUC: {roc_auc_balanced:.4f}")

# 3. С SMOTE (oversampling)
print("\n3. С SMOTE (oversampling):")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"   Размер после SMOTE: {X_train_smote.shape}")

gb_smote = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb_smote.fit(X_train_smote, y_train_smote)
y_pred_proba_smote = gb_smote.predict_proba(X_val)[:, 1]
roc_auc_smote = roc_auc_score(y_val, y_pred_proba_smote)
print(f"   ROC-AUC: {roc_auc_smote:.4f}")

# 4. С undersampling
print("\n4. С undersampling:")
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"   Размер после undersampling: {X_train_rus.shape}")

gb_rus = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb_rus.fit(X_train_rus, y_train_rus)
y_pred_proba_rus = gb_rus.predict_proba(X_val)[:, 1]
roc_auc_rus = roc_auc_score(y_val, y_pred_proba_rus)
print(f"   ROC-AUC: {roc_auc_rus:.4f}")

# Выбор лучшего метода
methods = {
    'base': roc_auc_base,
    'balanced': roc_auc_balanced,
    'smote': roc_auc_smote,
    'rus': roc_auc_rus
}

best_method = max(methods, key=methods.get)
print(f"\nЛучший метод: {best_method} (ROC-AUC: {methods[best_method]:.4f})")

# Обучение финальной модели на всех train данных с лучшим методом
print("\nОбучение финальной модели на всех данных...")

if best_method == 'smote':
    # Применяем SMOTE ко всем данным
    X_final, y_final = smote.fit_resample(X_scaled, y)
    print(f"Размер после SMOTE: {X_final.shape}")
elif best_method == 'rus':
    # Применяем undersampling ко всем данным
    X_final, y_final = rus.fit_resample(X_scaled, y)
    print(f"Размер после undersampling: {X_final.shape}")
else:
    # Используем исходные данные
    X_final, y_final = X_scaled, y

# Финальная модель с оптимизированными гиперпараметрами
final_model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    min_samples_split=50,
    min_samples_leaf=20
)

print("Обучение финальной модели...")
final_model.fit(X_final, y_final)

# Оценка на validation set
y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
roc_auc_final = roc_auc_score(y_val, y_val_pred_proba)
print(f"ROC-AUC на validation: {roc_auc_final:.4f}")

# Кросс-валидация
print("\nКросс-валидация...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_scaled, y, cv=cv, scoring='roc_auc')
print(f"Средний ROC-AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Анализ важности признаков
print("\nТоп-15 важных признаков:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(15))

# Анализ порогов для классификации
print("\nАнализ различных порогов классификации:")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for threshold in thresholds:
    y_pred_threshold = (y_val_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nПорог: {threshold}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# Предсказания на тестовых данных
print("\nГенерация предсказаний для тестовых данных...")
test_predictions = final_model.predict_proba(test_scaled)[:, 1]

# Создание submission файла
submission_df = pd.DataFrame({
    'front_id': test_df['front_id'],
    'target_value': test_predictions
})

submission_df.to_csv('improved_submission.csv', index=False)
print(f"Submission файл сохранен как 'improved_submission.csv'")
print(f"Размер submission: {submission_df.shape}")

# Статистика предсказаний
print(f"\nСтатистика предсказаний:")
print(f"  Минимальная вероятность: {test_predictions.min():.4f}")
print(f"  Максимальная вероятность: {test_predictions.max():.4f}")
print(f"  Средняя вероятность: {test_predictions.mean():.4f}")
print(f"  Медианная вероятность: {np.median(test_predictions):.4f}")

# Анализ распределения предсказаний
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(test_predictions, bins=bins)
print(f"\nРаспределение предсказаний по бинам:")
for i in range(len(bins)-1):
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} ({hist[i]/len(test_predictions)*100:.1f}%)")

print("\nМодель успешно обучена и предсказания сгенерированы!")