import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class CreditResponsePredictor:
    """Класс для предсказания вероятности согласия на кредит"""
    
    def __init__(self, model_path=None):
        """Инициализация предсказателя"""
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, client_data, is_training=False):
        """Подготовка признаков для модели"""
        
        # Создаем копию данных
        data = client_data.copy()
        
        # Удаляем идентификаторы если они есть
        if 'front_id' in data.columns:
            data = data.drop('front_id', axis=1)
        if 'decision_day' in data.columns:
            data = data.drop('decision_day', axis=1)
        
        # Сохраняем имена признаков
        if self.feature_names is None:
            self.feature_names = data.columns.tolist()
        
        # Кодирование категориальных признаков
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    # Для обучения создаем новый энкодер
                    if is_training:
                        self.label_encoders[col] = le
                        le.fit(data[col].astype(str))
                    else:
                        # Для предсказания используем сохраненный энкодер
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                        else:
                            # Если энкодер не найден, создаем новый
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            self.label_encoders[col] = le
                
                data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        # Обработка пропущенных значений
        if is_training:
            data_imputed = self.imputer.fit_transform(data)
        else:
            data_imputed = self.imputer.transform(data)
        
        # Масштабирование признаков
        if is_training:
            data_scaled = self.scaler.fit_transform(data_imputed)
        else:
            data_scaled = self.scaler.transform(data_imputed)
        
        return data_scaled
    
    def train(self, train_data, target_column='target_value'):
        """Обучение модели на данных"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        print("Подготовка данных для обучения...")
        X = train_data.drop(target_column, axis=1)
        y = train_data[target_column]
        
        # Подготовка признаков
        X_prepared = self.prepare_features(X, is_training=True)
        
        print("Обучение модели Gradient Boosting...")
        self.model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.7,
            min_samples_split=30,
            min_samples_leaf=10
        )
        
        self.model.fit(X_prepared, y)
        
        # Оценка модели
        from sklearn.metrics import roc_auc_score
        y_pred_proba = self.model.predict_proba(X_prepared)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        print(f"Модель обучена. ROC-AUC на train данных: {roc_auc:.4f}")
        
        # Анализ важности признаков
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            print("\nТоп-10 важных признаков:")
            print(feature_importance.head(10).to_string())
        
        return self
    
    def predict_proba(self, client_data):
        """Предсказание вероятности согласия на кредит"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала обучите модель или загрузите сохраненную.")
        
        # Подготовка признаков
        X_prepared = self.prepare_features(client_data, is_training=False)
        
        # Предсказание вероятности
        probabilities = self.model.predict_proba(X_prepared)[:, 1]
        
        return probabilities
    
    def predict(self, client_data, threshold=0.3):
        """Предсказание класса (согласие/отказ) с заданным порогом"""
        probabilities = self.predict_proba(client_data)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Сохранение модели и предобработчиков"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """Загрузка модели и предобработчиков"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"Модель загружена из {filepath}")
    
    def get_recommendation(self, probability, threshold=0.3):
        """Получение рекомендации на основе вероятности"""
        if probability >= threshold:
            return "РЕКОМЕНДУЕТСЯ: Высокая вероятность согласия на кредит"
        else:
            return "НЕ РЕКОМЕНДУЕТСЯ: Низкая вероятность согласия на кредит"

# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("ПРЕДСКАЗАТЕЛЬ ВЕРОЯТНОСТИ СОГЛАСИЯ НА КРЕДИТ")
    print("=" * 60)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    train_df = pd.read_csv('train_apps.csv')
    test_df = pd.read_csv('test_apps.csv')
    
    # Создание и обучение предсказателя
    predictor = CreditResponsePredictor()
    
    # Обучение модели (можно пропустить, если модель уже обучена)
    print("\nОбучение модели...")
    predictor.train(train_df)
    
    # Сохранение модели
    predictor.save_model('credit_response_model.pkl')
    
    # Пример предсказания для одного клиента из test данных
    print("\n" + "=" * 60)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ ДЛЯ КЛИЕНТА")
    print("=" * 60)
    
    # Берем первого клиента из test данных
    sample_client = test_df.iloc[[0]].copy()
    client_id = sample_client['front_id'].iloc[0]
    
    print(f"\nКлиент ID: {client_id}")
    
    # Предсказание вероятности
    probability = predictor.predict_proba(sample_client)[0]
    
    print(f"\nВероятность согласия на кредит: {probability:.4f} ({probability*100:.1f}%)")
    
    # Рекомендации с разными порогами
    print("\nРекомендации с разными порогами:")
    print("-" * 50)
    
    thresholds = {
        0.1: "Агрессивный подход",
        0.3: "Балансированный подход",
        0.5: "Консервативный подход"
    }
    
    for threshold, description in thresholds.items():
        recommendation = predictor.get_recommendation(probability, threshold)
        print(f"\n{description} (порог: {threshold}):")
        print(f"  {recommendation}")
    
    # Предсказание для всех test данных
    print("\n" + "=" * 60)
    print("ПРЕДСКАЗАНИЕ ДЛЯ ВСЕХ TEST ДАННЫХ")
    print("=" * 60)
    
    all_probabilities = predictor.predict_proba(test_df)
    
    # Создание submission файла
    submission_df = pd.DataFrame({
        'front_id': test_df['front_id'],
        'target_value': all_probabilities
    })
    
    submission_df.to_csv('model_predictions.csv', index=False)
    print(f"\nПредсказания сохранены в 'model_predictions.csv'")
    
    # Статистика предсказаний
    print(f"\nСтатистика предсказаний:")
    print(f"  Всего клиентов: {len(all_probabilities)}")
    print(f"  Средняя вероятность: {all_probabilities.mean():.4f}")
    print(f"  Медианная вероятность: {np.median(all_probabilities):.4f}")
    print(f"  Минимальная вероятность: {all_probabilities.min():.4f}")
    print(f"  Максимальная вероятность: {all_probabilities.max():.4f}")
    
    # Распределение по порогам
    print(f"\nРаспределение клиентов по вероятностям:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = (all_probabilities >= threshold).sum()
        percentage = count / len(all_probabilities) * 100
        print(f"  Вероятность ≥ {threshold}: {count} клиентов ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
    print("=" * 60)