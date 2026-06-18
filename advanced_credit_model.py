import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
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
print(f"Класс 0 (отказ): {class_dist[0]} ({class_dist[0]/len(train_df)*100:.1f}%)")
print(f"Класс 1 (согласие): {class_dist[1]} ({class_dist[1]/len(train_df)*100:.1f}%)")
print(f"Соотношение: 1:{class_dist[0]/class_dist[1]:.1f}")

# FEATURE ENGINEERING
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    """Создание новых признаков на основе имеющихся данных"""
    df = df.copy()
    
    # 1. Анализ соотношения предложенной ставки и ключевой ставки
    if 'offered_rate' in df.columns and 'cb_rate' in df.columns:
        df['rate_ratio'] = df['offered_rate'] / (df['cb_rate'] + 1e-10)
        df['rate_diff'] = df['offered_rate'] - df['cb_rate']
        df['rate_premium'] = (df['offered_rate'] - df['cb_rate']) / (abs(df['cb_rate']) + 1e-10)
    
    # 2. Учет соотношения запрошенного лимита и доступных лимитов
    if 'overdraft_limit_min' in df.columns and 'overdraft_limit_max' in df.columns:
        df['limit_range'] = df['overdraft_limit_max'] - df['overdraft_limit_min']
        df['limit_midpoint'] = (df['overdraft_limit_min'] + df['overdraft_limit_max']) / 2
        
        if 'loan_amount_last' in df.columns:
            df['loan_to_limit_min'] = df['loan_amount_last'] / (abs(df['overdraft_limit_min']) + 1e-10)
            df['loan_to_limit_max'] = df['loan_amount_last'] / (abs(df['overdraft_limit_max']) + 1e-10)
            df['loan_to_limit_mid'] = df['loan_amount_last'] / (abs(df['limit_midpoint']) + 1e-10)
    
    # 3. Признаки финансовой активности клиента
    # Суммарные признаки задолженностей
    debt_cols = ['sum_deb_ul_90', 'sum_deb_ul_30', 'sum_deb_investment_90']
    for col in debt_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(abs(df[col]) + 1e-10)
    
    # Признаки счетчиков
    count_cols = ['cnt_deb_loan_90', 'cnt_deb_ul_ip_90', 'cnt_deb_ul_ip_30', 'cnt_cred_loan_90']
    for col in count_cols:
        if col in df.columns:
            df[f'{col}_bin'] = (df[col] > 0).astype(int)
    
    # 4. Временные признаки из decision_day
    if 'decision_day' in df.columns:
        df['decision_day'] = pd.to_datetime(df['decision_day'])
        df['decision_month'] = df['decision_day'].dt.month
        df['decision_quarter'] = df['decision_day'].dt.quarter
        df['decision_dayofweek'] = df['decision_day'].dt.dayofweek
        df['decision_dayofyear'] = df['decision_day'].dt.dayofyear
    
    # 5. Признаки баланса и кредитных оборотов
    if 'balance_rur_amt_30_min' in df.columns:
        df['balance_abs'] = abs(df['balance_rur_amt_30_min'])
    
    if 'loan_rev_max_start_non_fin' in df.columns and 'loan_rev_min_start_fin' in df.columns:
        df['loan_rev_range'] = df['loan_rev_max_start_non_fin'] - df['loan_rev_min_start_fin']
        df['loan_rev_avg'] = (df['loan_rev_max_start_non_fin'] + df['loan_rev_min_start_fin']) / 2
    
    # 6. Признаки терминов кредитования
    if 'app_term_mean_360' in df.columns and 'overdraft_app_term_max_360' in df.columns:
        df['term_ratio'] = df['app_term_mean_360'] / (df['overdraft_app_term_max_360'] + 1e-10)
    
    # 7. Признаки активности в системе
    if 'count_all_corp_dashboard_events' in df.columns:
        df['events_log'] = np.log1p(abs(df['count_all_corp_dashboard_events']) + 1e-10)
    
    if 'p75_time_spent_minutes' in df.columns:
        df['time_spent_log'] = np.log1p(abs(df['p75_time_spent_minutes']) + 1e-10)
    
    # 8. Взаимодействия важных признаков
    if 'loan_amount_last' in df.columns and 'cnt_deb_loan_90' in df.columns:
        df['loan_debt_interaction'] = df['loan_amount_last'] * df['cnt_deb_loan_90']
    
    if 'loan_rev_max_start_non_fin' in df.columns and 'balance_rur_amt_30_min' in df.columns:
        df['rev_balance_interaction'] = df['loan_rev_max_start_non_fin'] * df['balance_rur_amt_30_min']
    
    return df

print("Создание новых признаков...")
train_df = create_features(train_df)
test_df = create_features(test_df)

print(f"Размер train после feature engineering: {train_df.shape}")
print(f"Размер test после feature engineering: {test_df.shape}")

# Подготовка данных для моделирования
print("\n" + "="*60)
print("ПОДГОТОВКА ДАННЫХ")
print("="*60)

# Разделение на признаки и целевую переменную
X = train_df.drop('target_value', axis=1)
y = train_df['target_value']

# Удаляем front_id и decision_day (после извлечения временных признаков)
X = X.drop(['front_id', 'decision_day'], axis=1, errors='ignore')
test_features = test_df.drop(['front_id', 'decision_day'], axis=1, errors='ignore')

# Сохраняем front_id для submission
test_front_ids = test_df['front_id']

# Обработка категориальных признаков
print("Обработка категориальных признаков...")
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Категориальные признаки: {list(categorical_cols)}")

# One-Hot Encoding для категориальных признаков с небольшим количеством уникальных значений
for col in categorical_cols:
    if col in X.columns:
        # Для признаков с небольшим количеством уникальных значений используем One-Hot Encoding
        if X[col].nunique() <= 10:
            # One-Hot Encoding
            X_encoded = pd.get_dummies(X[col], prefix=col, drop_first=True)
            test_encoded = pd.get_dummies(test_features[col], prefix=col, drop_first=True)
            
            # Выравнивание столбцов
            all_columns = set(X_encoded.columns) | set(test_encoded.columns)
            for col_name in all_columns:
                if col_name not in X_encoded.columns:
                    X_encoded[col_name] = 0
                if col_name not in test_encoded.columns:
                    test_encoded[col_name] = 0
            
            # Сортировка столбцов
            X_encoded = X_encoded[sorted(all_columns)]
            test_encoded = test_encoded[sorted(all_columns)]
            
            # Замена исходного признака
            X = pd.concat([X.drop(col, axis=1), X_encoded], axis=1)
            test_features = pd.concat([test_features.drop(col, axis=1), test_encoded], axis=1)
        else:
            # Для признаков с большим количеством уникальных значений используем Label Encoding
            le = LabelEncoder()
            combined = pd.concat([X[col], test_features[col]], axis=0)
            le.fit(combined.astype(str))
            X[col] = le.transform(X[col].astype(str))
            test_features[col] = le.transform(test_features[col].astype(str))

print(f"Размер X после обработки категориальных признаков: {X.shape}")
print(f"Размер test_features после обработки категориальных признаков: {test_features.shape}")

# Обработка пропущенных значений
print("\nОбработка пропущенных значений...")
# Используем KNNImputer для более точной импутации
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_features)

# Масштабирование признаков
print("Масштабирование признаков...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

# FEATURE SELECTION
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Используем Random Forest для отбора важных признаков
print("Отбор важных признаков с помощью Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_scaled, y)

# Выбираем признаки с важностью выше медианы
importances = rf_selector.feature_importances_
median_importance = np.median(importances[importances > 0])
selected_indices = importances > median_importance

X_selected = X_scaled[:, selected_indices]
test_selected = test_scaled[:, selected_indices]

print(f"Исходное количество признаков: {X_scaled.shape[1]}")
print(f"Количество отобранных признаков: {X_selected.shape[1]}")
print(f"Сокращение: {100*(1 - X_selected.shape[1]/X_scaled.shape[1]):.1f}%")

# Разделение на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмеры данных:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"Test: {test_selected.shape}")

# МОДЕЛИРОВАНИЕ
print("\n" + "="*60)
print("МОДЕЛИРОВАНИЕ")
print("="*60)

# Определяем модели для ансамбля
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced',
        C=0.1
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10
    )
}

# Обучение и оценка отдельных моделей
print("Обучение и оценка отдельных моделей...")
individual_results = {}

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train, y_train)
    
    # Предсказания на validation
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc', n_jobs=-1)
    
    individual_results[name] = {
        'model': model,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  ROC-AUC (val): {roc_auc:.4f}")
    print(f"  ROC-AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Создание ансамбля моделей
print("\nСоздание ансамбля моделей...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', models['Logistic Regression']),
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting'])
    ],
    voting='soft',
    weights=[1, 2, 3]  # Больший вес для Gradient Boosting
)

ensemble.fit(X_train, y_train)
y_pred_proba_ensemble = ensemble.predict_proba(X_val)[:, 1]
roc_auc_ensemble = roc_auc_score(y_val, y_pred_proba_ensemble)

print(f"ROC-AUC ансамбля: {roc_auc_ensemble:.4f}")

# Выбор лучшей модели
best_model_name = max(individual_results, key=lambda x: individual_results[x]['roc_auc'])
if roc_auc_ensemble > individual_results[best_model_name]['roc_auc']:
    best_model = ensemble
    best_model_name = 'Ensemble'
    best_roc_auc = roc_auc_ensemble
else:
    best_model = individual_results[best_model_name]['model']
    best_roc_auc = individual_results[best_model_name]['roc_auc']

print(f"\nЛучшая модель: {best_model_name}")
print(f"ROC-AUC: {best_roc_auc:.4f}")

# Подбор гиперпараметров для лучшей модели
print("\nПодбор гиперпараметров для Gradient Boosting...")
if best_model_name in ['Gradient Boosting', 'Ensemble']:
    param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Используем лучшую модель
    best_model = grid_search.best_estimator_

# Обучение финальной модели на всех данных
print("\nОбучение финальной модели на всех данных...")
final_model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    min_samples_split=20,
    min_samples_leaf=10
)

final_model.fit(X_selected, y)

# Кросс-валидация финальной модели
print("Кросс-валидация финальной модели...")
cv_scores_final = cross_val_score(final_model, X_selected, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Средний ROC-AUC (CV): {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std()*2:.4f})")

# Анализ важности признаков
print("\nТоп-20 важных признаков:")
feature_names = X.columns.tolist()
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]

feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(20).to_string())

# Предсказания на тестовых данных
print("\nГенерация предсказаний для тестовых данных...")
test_predictions = final_model.predict_proba(test_selected)[:, 1]

# Создание submission файла
submission_df = pd.DataFrame({
    'front_id': test_front_ids,
    'target_value': test_predictions
})

submission_df.to_csv('advanced_submission.csv', index=False)
print(f"Submission файл сохранен как 'advanced_submission.csv'")

# Статистика предсказаний
print(f"\nСтатистика предсказаний:")
print(f"  Всего клиентов: {len(test_predictions)}")
print(f"  Средняя вероятность: {test_predictions.mean():.6f}")
print(f"  Медианная вероятность: {np.median(test_predictions):.6f}")
print(f"  Минимальная вероятность: {test_predictions.min():.6f}")
print(f"  Максимальная вероятность: {test_predictions.max():.6f}")
print(f"  Стандартное отклонение: {test_predictions.std():.6f}")

# Распределение по порогам
print(f"\nРаспределение клиентов по вероятностям:")
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
for threshold in thresholds:
    count = (test_predictions >= threshold).sum()
    percentage = count / len(test_predictions) * 100
    print(f"  Вероятность ≥ {threshold}: {count} клиентов ({percentage:.1f}%)")

print("\n" + "="*60)
print("МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
print("="*60)
print("\nФайлы результатов:")
print("1. advanced_submission.csv - предсказания для отправки на платформу")
print("2. advanced_credit_model.py - скрипт с улучшенной моделью")
print("\nДля отправки на платформу используйте файл: advanced_submission.csv")