# Модель прогнозирования кредитного отклика

## Описание проекта
Модель машинного обучения для прогнозирования вероятности согласия клиента на кредитное предложение (бизнес-кредит, возобновляемая кредитная линия или овердрафт).

## Структура проекта

### Файлы данных
- `train_apps.csv` - обучающие данные (145,241 записей)
- `test_apps.csv` - тестовые данные (36,311 записей)
- `sample_submission.csv` - пример формата submission

### Основные скрипты
1. `credit_response_model.py` - базовая модель
2. `simplified_credit_model.py` - улучшенная модель с балансировкой классов
3. `predict_single_client.py` - класс для предсказаний с примером использования
4. `simple_visualization.py` - визуализация результатов

### Результаты
1. `final_submission.csv` - финальные предсказания
2. `model_predictions.csv` - предсказания от обученной модели
3. `credit_response_model.pkl` - сохраненная модель
4. `predictions_analysis.png` - графики анализа предсказаний
5. `target_distribution.png` - распределение классов
6. `model_report.md` - подробный отчет

## Быстрый старт

### Установка зависимостей
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Запуск модели
```bash
# 1. Обучение и предсказание
python3 simplified_credit_model.py

# 2. Визуализация результатов
python3 simple_visualization.py

# 3. Использование готовой модели
python3 predict_single_client.py
```

## Результаты модели

### Метрики качества
- **ROC-AUC**: 0.8258
- **Accuracy**: 0.9435
- **F1-score**: 0.2684 (из-за дисбаланса классов)

### Распределение предсказаний
- Средняя вероятность: 9.15%
- Медианная вероятность: 4.95%
- 24.0% клиентов имеют вероятность ≥ 10%
- 3.0% клиентов имеют вероятность ≥ 50%

### Важные признаки
1. `loan_rev_max_start_non_fin` - максимальный оборот по кредитам
2. `cnt_deb_loan_90` - количество задолженностей за 90 дней
3. `loan_amount_last` - последняя сумма кредита
4. `loan_rev_min_start_fin` - минимальный оборот по кредитам
5. `balance_rur_amt_30_min` - минимальный баланс за 30 дней

## Использование модели

### Для batch-предсказаний
```python
import pandas as pd
from predict_single_client import CreditResponsePredictor

# Загрузка данных
data = pd.read_csv('test_apps.csv')

# Создание предсказателя
predictor = CreditResponsePredictor('credit_response_model.pkl')

# Предсказание вероятностей
probabilities = predictor.predict_proba(data)

# Создание submission
submission = pd.DataFrame({
    'front_id': data['front_id'],
    'target_value': probabilities
})
submission.to_csv('predictions.csv', index=False)
```

### Для одного клиента
```python
from predict_single_client import CreditResponsePredictor

# Загрузка модели
predictor = CreditResponsePredictor('credit_response_model.pkl')

# Данные клиента (пример)
client_data = pd.DataFrame({
    'loan_amount_last': [1.5],
    'overdraft_limit_min': [-1.8],
    # ... остальные признаки
})

# Предсказание
probability = predictor.predict_proba(client_data)[0]
print(f"Вероятность согласия: {probability:.1%}")

# Рекомендация
if probability >= 0.3:
    print("РЕКОМЕНДУЕТСЯ: Высокая вероятность согласия")
else:
    print("НЕ РЕКОМЕНДУЕТСЯ: Низкая вероятность согласия")
```

## Рекомендации по порогам

### Бизнес-стратегии
1. **Агрессивная (порог 0.1)**
   - 24.0% клиентов
   - Максимальный охват, выше риск дефолта

2. **Балансированная (порог 0.3)**
   - 5.9% клиентов
   - Оптимальное соотношение риск/доходность

3. **Консервативная (порог 0.5)**
   - 3.0% клиентов
   - Минимальный риск, высокая вероятность возврата

## Ограничения модели

### Технические ограничения
1. **Дисбаланс классов**: 6.1% положительных ответов
2. **Пропущенные значения**: многие признаки имеют >50% пропусков
3. **Ограниченные признаки**: отсутствуют данные о доходе, кредитной истории из БКИ

### Бизнес-ограничения
1. Модель предсказывает вероятность, но не сумму кредита
2. Не учитывает макроэкономические факторы
3. Требует регулярного переобучения на новых данных
