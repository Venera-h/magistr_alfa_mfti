import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
print("Загрузка данных...")
submission = pd.read_csv('final_submission.csv')
train_data = pd.read_csv('train_apps.csv')

print(f"Размер submission: {submission.shape}")
print(f"Размер train данных: {train_data.shape}")

# Анализ распределения предсказаний
predictions = submission['target_value']

# Создание графиков
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Анализ предсказаний модели кредитного отклика', fontsize=16, fontweight='bold')

# 1. Гистограмма распределения предсказаний
axes[0, 0].hist(predictions, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(x=predictions.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {predictions.mean():.3f}')
axes[0, 0].axvline(x=np.median(predictions), color='green', linestyle='--', linewidth=2, label=f'Медиана: {np.median(predictions):.3f}')
axes[0, 0].set_xlabel('Вероятность согласия')
axes[0, 0].set_ylabel('Количество клиентов')
axes[0, 0].set_title('Распределение предсказанных вероятностей')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot предсказаний
axes[0, 1].boxplot(predictions, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[0, 1].set_xlabel('Вероятность согласия')
axes[0, 1].set_title('Box plot предсказаний')
axes[0, 1].grid(True, alpha=0.3)

# 3. Кумулятивное распределение
sorted_pred = np.sort(predictions)
cumulative = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
axes[1, 0].plot(sorted_pred, cumulative, linewidth=2, color='purple')
axes[1, 0].set_xlabel('Вероятность согласия')
axes[1, 0].set_ylabel('Доля клиентов')
axes[1, 0].set_title('Кумулятивное распределение вероятностей')
axes[1, 0].grid(True, alpha=0.3)

# Добавление перцентилей
percentiles = [10, 25, 50, 75, 90, 95]
for p in percentiles:
    value = np.percentile(predictions, p)
    axes[1, 0].axvline(x=value, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].text(value, 0.1, f'{p}%', rotation=90, fontsize=9)

# 4. Распределение по порогам
thresholds = np.arange(0, 1.1, 0.1)
counts_above = [(predictions >= t).sum() for t in thresholds]
percentages = [count/len(predictions)*100 for count in counts_above]

bars = axes[1, 1].bar(thresholds, percentages, width=0.08, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].set_xlabel('Порог вероятности')
axes[1, 1].set_ylabel('Процент клиентов выше порога (%)')
axes[1, 1].set_title('Процент клиентов выше различных порогов')
axes[1, 1].grid(True, alpha=0.3)

# Добавление значений на столбцы
for bar, percentage in zip(bars, percentages):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
print("Графики сохранены в 'predictions_analysis.png'")

# Анализ распределения в train данных
print("\nАнализ распределения целевой переменной в train данных:")
target_dist = train_data['target_value'].value_counts()
print(f"Класс 0 (отказ): {target_dist[0]} ({target_dist[0]/len(train_data)*100:.1f}%)")
print(f"Класс 1 (согласие): {target_dist[1]} ({target_dist[1]/len(train_data)*100:.1f}%)")

# Создание графика распределения классов
fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = ['#ff6b6b', '#51cf66']
bars = ax2.bar(['Отказ (0)', 'Согласие (1)'], target_dist.values, color=colors, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Количество записей')
ax2.set_title('Распределение целевой переменной в обучающих данных')
ax2.grid(True, alpha=0.3, axis='y')

# Добавление значений на столбцы
for bar, count in zip(bars, target_dist.values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
            f'{count:,}', ha='center', va='bottom', fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{count/len(train_data)*100:.1f}%', ha='center', va='center', 
            color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("График распределения классов сохранен в 'target_distribution.png'")

# Создание отчета по порогам
print("\nОтчет по порогам классификации:")
print("-" * 60)
print(f"{'Порог':<10} {'Кол-во клиентов':<20} {'Процент':<15} {'Описание'}")
print("-" * 60)

threshold_descriptions = {
    0.1: "Агрессивный подход (максимальный охват)",
    0.2: "Умеренно-агрессивный",
    0.3: "Балансированный",
    0.4: "Умеренно-консервативный",
    0.5: "Консервативный (минимальный риск)",
    0.6: "Очень консервативный",
    0.7: "Сверхконсервативный"
}

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    count = (predictions >= threshold).sum()
    percentage = count / len(predictions) * 100
    description = threshold_descriptions.get(threshold, "")
    print(f"{threshold:<10} {count:<20,} {percentage:<15.1f}% {description}")

print("-" * 60)

# Статистика предсказаний
print("\nСтатистика предсказаний:")
stats = {
    'Минимум': predictions.min(),
    '1%-й перцентиль': np.percentile(predictions, 1),
    '10%-й перцентиль': np.percentile(predictions, 10),
    '25%-й перцентиль': np.percentile(predictions, 25),
    'Медиана': np.median(predictions),
    '75%-й перцентиль': np.percentile(predictions, 75),
    '90%-й перцентиль': np.percentile(predictions, 90),
    '99%-й перцентиль': np.percentile(predictions, 99),
    'Максимум': predictions.max(),
    'Среднее': predictions.mean(),
    'Стандартное отклонение': predictions.std()
}

for stat, value in stats.items():
    print(f"{stat:<25} {value:.6f}")

print("\nВизуализация завершена!")
print("\nФайлы результатов:")
print("1. final_submission.csv - предсказания вероятностей")
print("2. predictions_analysis.png - графики анализа предсказаний")
print("3. target_distribution.png - распределение классов")
print("4. model_report.md - подробный отчет по модели")