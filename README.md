# Airbnb Price Predictor

Sistema de predicción de precios para alquileres de Airbnb en Buenos Aires utilizando regresión lineal, Ridge y Lasso.

## Estructura del Proyecto

```
airbnb-price-predictor/
├── configs/
│   └── config.yaml          # Configuración de hiperparámetros y features
├── data/
│   ├── train.csv           # Datos de entrenamiento
│   └── test.csv            # Datos de prueba
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Carga y división de datos
│   ├── feature_engineering.py  # Ingeniería de features
│   ├── transformations.py  # Transformaciones de datos
│   └── models.py           # Entrenamiento y evaluación de modelos
├── notebooks/
│   └── (notebooks originales para referencia)
├── models/                  # Modelos guardados
├── predictions/             # Predicciones generadas
├── train.py                # Script principal de entrenamiento
├── train_ablation.py       # Script para ablation study de features
└── requirements.txt        # Dependencias del proyecto
```

## Características

### Features Implementadas

- **Distancia al centro**: Distancia euclidiana desde el centro de Buenos Aires
- **Features temporales**: Días, semanas, meses, trimestres y años desde la última review
- **Categorías de noches mínimas**: Clasificación en corta, semana, mes, semestre, largo plazo
- **Host con múltiples listados**: Flag booleano
- **Ratio de reviews**: Reviews por día

### Modelos

1. **Regresión Lineal**: Baseline con y sin scaling
2. **Ridge Regression**: Regularización L2 (alpha=1.0)
3. **Lasso Regression**: Regularización L1 para selección de features (alpha=1.0)
4. **Decision Tree**: Árbol de decisión con GridSearchCV
5. **Random Forest**: Ensemble de árboles con GridSearchCV
6. **Gradient Boosting**: Boosting con GridSearchCV
7. **Neural Network (MLP)**: Red neuronal con GridSearchCV

## Instalación

### Requisitos

- Python 3.8+
- pip

### Setup

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/airbnb-price-predictor.git
cd airbnb-price-predictor
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Agregar tus datos:
```bash
# Copiar train.csv y test.csv a la carpeta data/
cp /ruta/a/tus/datos/train.csv data/
cp /ruta/a/tus/datos/test.csv data/
```

## Uso

### Entrenar Modelos

```bash
python train.py
```

El script hará lo siguiente:
1. Cargará los datos de entrenamiento y prueba
2. Aplicará feature engineering según la configuración
3. Entrenará los tres modelos (Linear, Ridge, Lasso)
4. Evaluará en conjuntos de train, validation y test
5. Generará predicciones para el conjunto de prueba

### Configuración

Edita `configs/config.yaml` para ajustar:
- Rutas de datos
- Features a utilizar
- Hiperparámetros de modelos
- Parámetros de transformación

Ejemplo de configuración:

```yaml
models:
  ridge:
    alpha: 1.0  # Cambia este valor para ajustar la regularización
    use_scaling: true

  lasso:
    alpha: 10.0
    max_iter: 10000
```

## Resultados

Los resultados incluyen las siguientes métricas:
- **MAE** (Mean Absolute Error): Error promedio en dólares
- **RMSE** (Root Mean Squared Error): Error cuadrático medio
- **R²** (Coeficiente de determinación): Qué tan bien el modelo explica la variabilidad

Las predicciones se guardan en:
- `predictions/predictions_linear_regression_unscaled.csv`
- `predictions/predictions_linear_regression_scaled.csv`
- `predictions/predictions_ridge.csv`
- `predictions/predictions_lasso.csv`
- `predictions/predictions_decision_tree.csv`
- `predictions/predictions_random_forest.csv`
- `predictions/predictions_gradient_boosting.csv`
- `predictions/predictions_neural_network.csv`
- `predictions/model_comparison.csv` - Tabla comparativa de todos los modelos

## Feature Ablation Study

El proyecto incluye un sistema para entrenar modelos con diferentes combinaciones de grupos de features, permitiendo analizar qué features contribuyen más a la predicción.

### Grupos de Features

| Grupo | Features |
|-------|----------|
| **location** | latitude, longitude, distance_to_center, neighbourhood (one-hot) |
| **property** | room_type (one-hot), minimum_nights, minimum_nights_num |
| **host** | calculated_host_listings_count, host_has_multiple_listings |
| **reviews** | number_of_reviews, reviews_per_month, reviews_ratio |
| **time** | last_review, days/weeks/months/quarters/years_since_last_review |

### Uso del Ablation Study

```bash
# Ejecutar con combinaciones predefinidas en config
python train_ablation.py

# Probar todas las combinaciones posibles (2^5 - 1 = 31 combinaciones)
python train_ablation.py --all-combinations

# Probar grupos específicos
python train_ablation.py --groups location property reviews

# Especificar modelos a evaluar
python train_ablation.py --models random_forest gradient_boosting neural_network

# Usar grid search para optimización de hiperparámetros
python train_ablation.py --grid-search

# Combinar opciones
python train_ablation.py --all-combinations --models random_forest --grid-search --output results/ablation.csv
```

### Salida

El script genera:
- `predictions/ablation_results.csv` - CSV con todos los resultados
- Tablas resumen mostrando mejor R² por combinación y por modelo
- Recomendación de la mejor configuración encontrada

### Configuración en config.yaml

```yaml
# Grupos de features para ablation study
feature_groups:
  location:
    enabled: true
    columns:
      - latitude
      - longitude
      - distance_to_center
      - neighbourhood
  property:
    enabled: true
    columns:
      - room_type
      - minimum_nights
      - minimum_nights_num
  # ... más grupos

# Configuración del ablation study
ablation:
  enabled: false
  models:
    - linear_regression
    - random_forest
    - gradient_boosting
  combinations:
    - ["all"]
    - ["location", "property", "host"]
    - ["property", "host", "reviews", "time"]
```

## Mejoras Futuras

- [x] Grid search para optimización de hiperparámetros
- [x] Validación cruzada (K-Fold, Repeated K-Fold)
- [x] Modelos más complejos (Random Forest, Gradient Boosting, Neural Networks)
- [x] Feature ablation study
- [ ] Feature selection automático
- [ ] Análisis de features importantes
- [ ] Dashboard de visualización

## Comparación con Colab

### Ventajas de esta versión local:

✅ **Modular**: Código separado en módulos reutilizables
✅ **Configurable**: Fácil ajuste de hiperparámetros sin tocar código
✅ **Reproducible**: Configuración en YAML, seeds fijados
✅ **Mantenible**: Mejor organización y documentación
✅ **Versionable**: Ideal para Git
✅ **Extensible**: Fácil agregar nuevos modelos o features

## Contribuir

Si quieres mejorar el proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -am 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

## Licencia

MIT

## Autores

Proyecto creado como migración de notebook de Colab a código local estructurado.
