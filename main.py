from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = load_iris()
X = data.data
y = data.target

# Modelo de bosque aleatorio con hiperparámetros por defecto
model_default = RandomForestClassifier(random_state=0)

# Medición del tiempo y rendimiento con hiperparámetros por defecto
start_time_default = time.time()
scores_default = cross_val_score(model_default, X, y, cv=5)
end_time_default = time.time()
time_default = end_time_default - start_time_default
mean_score_default = np.mean(scores_default)

# Espacio de hiperparámetros para hyperopt
space = {
    'n_estimators': hp.choice('n_estimators', range(10, 150)),
    'max_depth': hp.choice('max_depth', range(1, 30)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 10))
}

# Función objetivo para hyperopt
def objective(params):
    model = RandomForestClassifier(**params, random_state=0)
    score = cross_val_score(model, X, y, cv=5).mean()
    return {'loss': -score, 'status': STATUS_OK}

# Realizar la optimización con hyperopt
trials = Trials()
start_time_opt = time.time()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)
end_time_opt = time.time()
time_optimization = end_time_opt - start_time_opt

# Entrenar y evaluar el modelo con hiperparámetros optimizados
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = max(2, int(best_params['min_samples_split']))

model_optimized = RandomForestClassifier(**best_params, random_state=0)
start_time_opt_model = time.time()
scores_optimized = cross_val_score(model_optimized, X, y, cv=5)
end_time_opt_model = time.time()
time_opt_model = end_time_opt_model - start_time_opt_model
mean_score_optimized = np.mean(scores_optimized)

# Resultados
print("Tiempo de entrenamiento (default):", time_default)
print("Precisión media (default):", mean_score_default)
print("Tiempo de optimización:", time_optimization)
print("Tiempo de entrenamiento (optimizado):", time_opt_model)
print("Precisión media (optimizada):", mean_score_optimized)

# Gráfico de barras para comparar
labels = ['Default', 'Optimized']
time_values = [time_default, time_opt_model]
score_values = [mean_score_default, mean_score_optimized]

x = np.arange(len(labels))  # Ubicaciones de las etiquetas
width = 0.35  # Ancho de las barras

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, time_values, width, label='Tiempo')
rects2 = ax.bar(x + width/2, score_values, width, label='Precisión')

# Añadir texto para etiquetas, título y etiquetas personalizadas de los ejes
ax.set_ylabel('Valores')
ax.set_title('Comparación entre modelo por defecto y optimizado')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

# Gráficas individuales para cada modelo
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(scores_default, label='Default', marker='o')
plt.title('Rendimiento del Modelo Default')
plt.xlabel('Número de Validación')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(scores_optimized, label='Optimized', marker='o')
plt.title('Rendimiento del Modelo Optimizado')
plt.xlabel('Número de Validación')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()
