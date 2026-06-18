import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
train_df = pd.read_csv('train_apps.csv')
test_df = pd.read_csv('test_apps.csv')

print(f"Размер train данных: {train_df.shape}")
print(f"Размер test данных: {test_df.shape}")

# Анализ целевой переменной
print("\nАнализ целевой переменной:")
class_dist = train_df['target_value'].value_counts()
print(f"Класс 0: {class_dist[0]} ({class_dist[0]/len(train_df)*100:.1f}%)")
print(f"Класс 1: {class_dist[1]} ({class_dist[1]/len(train_df)*100:.1f}%)")
print(f"Соотношение: 1:{class_dist[0]/class_dist[1]:.1f}")

# Подготовка данных
print("\nПодготовка данных...")
X = train_df.drop('target_value', axis=1)
y = train_df['target_value']

# Удаляем идентификаторы
X = X.drop(['front_id', 'decision_day'], axis=1, errors='ignore')
test_features = test_df.drop(['front_id', 'decision_day'], axis=1, errors='ignore')

# Кодирование категориальных признаков
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Категориальные признаки: {list(categorical_cols)}")

label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        combined = pd.concat([X[col], test_features[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        test_features[col] = le.transform(test_features[col].astype(str))
        label_encoders[col] = le

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

# Обучение моделей с разными гиперпараметрами
print("\nОбучение моделей...")

models_config = [
    {
        'name': 'Gradient Boosting (базовый)',
        'model': GradientBoostingClassifier(random_state=42, n_estimators=100)
    },
    {
        'name': 'Gradient Boosting (с балансировкой)',
        'model': GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            min_samples_split=50,
            min_samples_leaf=20
        )
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced',
            max_depth=10
        )
    }
]

results = []
for config in models_config:
    print(f"\nОбучение {config['name']}...")
    model = config['model']
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Матрица ошибок
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'name': config['name'],
        'model': model,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })
    
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Выбор лучшей модели по ROC-AUC
best_result = max(results, key=lambda x: x['roc_auc'])
best_model = best_result['model']

print(f"\nЛучшая модель: {best_result['name']}")
print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"F1-score: {best_result['f1']:.4f}")

# Кросс-валидация лучшей модели
print("\nКросс-валидация лучшей модели...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='roc_auc')
print(f"Средний ROC-AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Анализ важности признаков
if hasattr(best_model, 'feature_importances_'):
    print("\nТоп-15 важных признаков:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance.head(15).to_string())

# Обучение финальной модели на всех данных
print("\nОбучение финальной модели на всех данных...")
final_model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.7,
    min_samples_split=30,
    min_samples_leaf=10
)

final_model.fit(X_scaled, y)

# Предсказания на тестовых данных
print("Генерация предсказаний для тестовых данных...")
test_predictions = final_model.predict_proba(test_scaled)[:, 1]

# Создание submission файла
submission_df = pd.DataFrame({
    'front_id': test_df['front_id'],
    'target_value': test_predictions
})

submission_df.to_csv('final_submission.csv', index=False)
print(f"Submission файл сохранен как 'final_submission.csv'")

# Анализ предсказаний
print(f"\nСтатистика предсказаний:")
print(f"  Минимум: {test_predictions.min():.6f}")
print(f"  Максимум: {test_predictions.max():.6f}")
print(f"  Среднее: {test_predictions.mean():.6f}")
print(f"  Медиана: {np.median(test_predictions):.6f}")
print(f"  Стандартное отклонение: {test_predictions.std():.6f}")

# Распределение предсказаний
print("\nРаспределение предсказаний:")
percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    value = np.percentile(test_predictions, p)
    print(f"  {p}%-й перцентиль: {value:.6f}")

# Количество предсказаний выше различных порогов
print("\nКоличество предсказаний выше порогов:")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    count = (test_predictions >= threshold).sum()
    percentage = count / len(test_predictions) * 100
    print(f"  >={threshold}: {count} ({percentage:.1f}%)")

print("\nМодель успешно обучена!")