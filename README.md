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

1. **Regresión Lineal**: Baseline con scaling
2. **Ridge Regression**: Regularización L2 (alpha=1.0)
3. **Lasso Regression**: Regularización L1 para selección de features (alpha=10.0)

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
- `predictions/predictions_linear_regression.csv`
- `predictions/predictions_ridge.csv`

## Mejoras Futuras

- [ ] Grid search para optimización de hiperparámetros
- [ ] Validación cruzada
- [ ] Modelos más complejos (Random Forest, XGBoost)
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
