# Proyecto Final MLOps

# Integrantes

* Mateo Ruiz Mendoza

* Carlos Manuel Carvajales

## Video

Link: [Video de sustentación](https://youtu.be/9-EUVbKnS4U)

## Sistema de Entrenamiento, Despliegue y Monitoreo de Modelos de Predicción de Precios de Vivienda

## Descripción General

Este proyecto implementa una plataforma completa de MLOps para el entrenamiento, despliegue, monitoreo y operación continua de modelos de Machine Learning orientados a la predicción de precios de vivienda.

La solución integra:

* Apache Airflow para orquestación.
* MLflow para experiment tracking y model registry.
* MinIO como almacenamiento de artefactos.
* MySQL como base de datos operacional.
* FastAPI para inferencia.
* Streamlit para interacción con usuarios.
* Prometheus para monitoreo.
* Grafana para visualización de métricas.
* Locust para pruebas de carga.
* Kubernetes para despliegue.
* ArgoCD para GitOps.

El pipeline implementa mecanismos automáticos de detección de drift, reentrenamiento, promoción de modelos y monitoreo operacional.

---

# Arquitectura General

La solución se compone de los siguientes módulos:

* Obtención de datos.
* Preprocesamiento.
* Detección de drift.
* Entrenamiento automático.
* Registro de modelos en MLflow.
* Promoción automática a Champion.
* Servicio de inferencia.
* Observabilidad.
* Pruebas de carga.
* Despliegue automatizado mediante GitOps.

---

# Estructura del Proyecto

```text
.
├── airflow/
├── argocd/
├── grafana/
├── inference_api/
├── k8s/
├── locust/
├── mlflow_compose/
├── mysql_db/
├── prometheus/
├── streamlit_ui/
├── test_notebooks/
├── docker-compose.yml
├── docker-compose-airflow.yml
├── docker-compose-k8s.yml
└── README.md
```

---

# Componentes Principales

## Airflow

Responsable de la ejecución del pipeline MLOps:

* Ingesta de datos.
* Preprocesamiento.
* Detección de drift.
* Entrenamiento.
* Registro de modelos.
* Promoción de Champion.

Código principal:

```text
airflow/dags/house_price.py
airflow/src/utils.py
```

---

## MLflow

Gestiona:

* Experimentos.
* Métricas.
* Parámetros.
* Artefactos.
* Versionado de modelos.
* Registro Champion/Challenger.

Durante cada entrenamiento se almacenan:

* Lotes utilizados.
* Commit asociado.
* Razón de entrenamiento.
* Métricas.
* Modelo serializado.
* Metadata de entrenamiento.

---

## MinIO

Almacenamiento de:

* Artefactos MLflow.
* Modelos.
* Logs del pipeline.

Bucket principal:

```text
pipeline-logs
```

---

## API de Inferencia

Implementada con FastAPI.

Funcionalidades:

* Predicción.
* Validación de payload.
* Exposición de métricas Prometheus.
* Registro de inferencias.
* Recarga automática del modelo Champion.

Archivo principal:

```text
inference_api/main.py
```

---

## Streamlit

Interfaz gráfica para:

* Generar payloads de ejemplo.
* Ejecutar inferencias.
* Visualizar resultados.
* Consultar logs almacenados en MinIO.

Archivos:

```text
streamlit_ui/app.py
streamlit_ui/pages/logs_page.py
```

---

## Prometheus

Recolecta métricas de:

* API de inferencia.
* Node Exporter.
* Kubernetes.
* Servicios del ecosistema.

Configuración:

```text
prometheus/prometheus.yml
```

---

## Grafana

Dashboards para:

* Consumo CPU.
* Consumo memoria.
* Estado del modelo.
* Latencia.
* Throughput.
* Errores.
* Métricas de inferencia.

Dashboards:

```text
grafana/dashboards/
```

---

## Locust

Pruebas de carga sobre la API de inferencia.

Archivo principal:

```text
locust/locustfile.py
```

---

# Servicios y Puertos

| Servicio       | Puerto |
| -------------- | ------ |
| Airflow        | 8080   |
| API Inferencia | 8001   |
| API Datos      | 8003   |
| Streamlit      | 8501   |
| MLflow         | 5000   |
| MinIO          | 9000   |
| Prometheus     | 9090   |
| Grafana        | 3000   |
| Locust         | 8089   |

---

# Ejecución Local con Docker Compose

## Levantar Infraestructura

```bash
docker compose up -d
```

o

```bash
docker-compose up -d
```

---

## Verificar Servicios

```bash
docker ps
```

---

## Acceso a Componentes

| Servicio       | URL                   |
| -------------- | --------------------- |
| Airflow        | http://localhost:8080 |
| API Inferencia | http://localhost:8001 |
| API Datos      | http://localhost:8003 |
| Streamlit      | http://localhost:8501 |
| MLflow         | http://localhost:5000 |
| MinIO          | http://localhost:9000 |
| Prometheus     | http://localhost:9090 |
| Grafana        | http://localhost:3000 |
| Locust         | http://localhost:8089 |

---

# Pipeline de Entrenamiento

El DAG principal es:

```text
house_price_pipeline
```

El flujo ejecuta:

1. Obtención de datos.
2. Preprocesamiento.
3. Cálculo de estadísticas.
4. Detección de drift.
5. Decisión de entrenamiento.
6. Entrenamiento.
7. Registro en MLflow.
8. Evaluación Champion vs Challenger.
9. Promoción automática.

---

# Observabilidad

## Métricas de Infraestructura

Recolectadas mediante:

* Node Exporter
* Prometheus

Incluyen:

* CPU.
* Memoria.
* Disponibilidad.

---

## Métricas de Modelo

Incluyen:

* Predicciones.
* Latencia.
* Errores.
* Modelo Champion activo.

---

## Dashboards

Grafana contiene dashboards preconfigurados para:

* Infraestructura.
* Inferencia.
* Modelo.
* Pruebas de carga.

---

# Despliegue en Kubernetes

## Arquitectura de Despliegue

La solución fue desplegada sobre un clúster local de Kubernetes utilizando Minikube como plataforma de orquestación.

El clúster fue configurado con:

```text
14 GB RAM
```

permitiendo ejecutar simultáneamente:

* Apache Airflow
* API de Obtención de Datos
* API de Inferencia
* MLflow
* MinIO
* MySQL
* Prometheus
* Grafana
* Streamlit
* Locust

---

## Organización de los Manifiestos

```text
k8s/
├── airflow/
├── api_inference/
├── get_data_api/
├── grafana/
├── locust/
├── minio/
├── mlflow/
├── mysql_db/
├── prometheus/
├── streamlit/
└── project-secrets.yml
```

Cada servicio incluye:

* Deployments
* Services
* PVC
* ConfigMaps
* Jobs de inicialización

---

## Aplicación de Recursos

```bash
kubectl apply -f k8s/project-secrets.yml

kubectl apply -f k8s/mysql_db/
kubectl apply -f k8s/minio/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/api_inference/
kubectl apply -f k8s/get_data_api/
kubectl apply -f k8s/prometheus/
kubectl apply -f k8s/grafana/
kubectl apply -f k8s/streamlit/
kubectl apply -f k8s/locust/
kubectl apply -f k8s/airflow/
```

---

## Verificación

```bash
kubectl get pods

kubectl get svc

kubectl get pvc
```

---

# Estrategia GitOps con ArgoCD

La sincronización continua fue implementada mediante ArgoCD.

La configuración principal se encuentra en:

```text
argocd/application-local-api.yaml
```

El repositorio Git se utiliza como única fuente de verdad para el estado deseado de la infraestructura.

---

# Integración Continua

El proyecto implementa CI mediante GitHub Actions.

Workflow:

```text
.github/workflows/main.yml
```

Cada push a la rama principal:

1. Ejecuta el pipeline CI.
2. Construye imágenes Docker.
3. Publica imágenes en Docker Hub.
4. Actualiza manifiestos Kubernetes.
5. Envía cambios al repositorio.

---

# Despliegue Continuo

ArgoCD:

1. Detecta cambios en Git.
2. Compara estado actual vs deseado.
3. Ejecuta sincronización automática.
4. Kubernetes descarga nuevas imágenes.
5. Realiza rollout automático.

Esto implementa una estrategia GitOps completa.

---



# Decisiones de Diseño

La arquitectura fue diseñada siguiendo principios de desacoplamiento, trazabilidad y automatización del ciclo de vida de modelos.

Las principales decisiones de diseño fueron:

### Orquestación mediante Airflow

Se seleccionó Apache Airflow como motor de orquestación debido a su capacidad para:

* Modelar pipelines complejos mediante DAGs.
* Programar ejecuciones periódicas.
* Implementar lógica condicional mediante Branch Operators.
* Integrarse fácilmente con MLflow y servicios externos.

### MLflow como Registro Centralizado

MLflow fue utilizado para centralizar:

* Experimentos.
* Versiones de modelos.
* Métricas.
* Artefactos.
* Registro Champion/Challenger.

Esto permite mantener trazabilidad completa entre:

* Datos utilizados.
* Código ejecutado.
* Modelo generado.
* Métricas obtenidas.

### Separación entre Entrenamiento e Inferencia

La API de inferencia fue desacoplada completamente del pipeline de entrenamiento.

Esto permite:

* Actualizar modelos sin reiniciar la infraestructura.
* Escalar inferencia independientemente.
* Aplicar estrategias Champion/Challenger.

La API consulta periódicamente MLflow para identificar si existe una nueva versión Champion y recargarla automáticamente.

### Persistencia de Artefactos

Todos los artefactos de entrenamiento son almacenados en MinIO mediante la integración nativa de MLflow.

Esto incluye:

* Modelos serializados.
* Metadata de entrenamiento.
* Métricas.
* Reportes.
* Gráficos.

### Observabilidad desde el Diseño

La observabilidad fue considerada como un componente central de la arquitectura.

Se instrumentaron métricas para:

* Infraestructura.
* API de inferencia.
* Modelo desplegado.
* Pruebas de carga.

Las métricas son recolectadas por Prometheus y visualizadas mediante Grafana.

---

# Criterios de Entrenamiento

El sistema implementa una estrategia de reentrenamiento automático basada en detección de drift sobre los datos.

La decisión de entrenamiento se realiza mediante la tarea:

```text
decide_training
```

del DAG principal.

## Primer Entrenamiento

Si no existe un modelo Champion registrado en MLflow:

```text
Champion = None
```

el sistema fuerza automáticamente el entrenamiento de un nuevo modelo.

## Drift Numérico

Para las variables numéricas:

```text
price
bed
bath
acre_lot
house_size
```

se comparan:

* Media histórica.
* Desviación estándar histórica.

contra las estadísticas del lote actual.

Se calcula:

```text
Mean Drift
Std Drift
```

y se considera drift cuando la diferencia relativa supera:

```text
25%
```

Configuración actual:

```text
DRIFT_THRESHOLD = 0.25
MIN_NUMERIC_DRIFTS = 2
```

Por lo tanto, el entrenamiento se activa cuando al menos dos mediciones de drift superan el umbral configurado.

## Drift Categórico

Para las variables:

```text
status
city
state
zip_code
```

se detectan categorías no observadas durante el entrenamiento del Champion.

Si aparecen nuevas categorías:

```text
unseen categories
```

el sistema considera que existe drift categórico y activa el entrenamiento.

## Razón de Entrenamiento

Cuando se decide entrenar, el sistema genera automáticamente una descripción detallada de la decisión.

Ejemplo:

```text
Current batch=15

Reference batches=[1,2,3]

price: mean drift 34.2%

house_size: std drift 29.5%
```

Esta información es almacenada en MLflow como artefacto del experimento para mantener trazabilidad de la decisión tomada.

---

# Registro de Información de Entrenamiento

Cada ejecución de entrenamiento registra en MLflow:

### Información del Dataset

* Batch utilizado.
* Conjunto de batches históricos.
* Cantidad de observaciones.
* Variables utilizadas.

### Información del Código

* Commit Git asociado al entrenamiento.
* Fecha y hora de ejecución.

### Parámetros del Modelo

Incluyendo:

```text
iterations
learning_rate
depth
loss_function
random_seed
```

y demás hiperparámetros relevantes.

### Métricas

Entrenamiento:

* RMSE
* MAE
* R²

Validación:

* RMSE
* MAE
* R²

Prueba:

* RMSE
* MAE
* R²

### Artefactos

Entre los artefactos almacenados se incluyen:

* Modelo serializado.
* Metadata de entrenamiento.
* Razón de entrenamiento.
* Reportes de evaluación.
* Gráficos de desempeño.

---

# Reglas de Promoción Champion / Challenger

El proyecto implementa una estrategia automática Champion/Challenger utilizando MLflow Model Registry.

## Champion

Corresponde al modelo actualmente desplegado en producción.

Se encuentra registrado con el alias:

```text
champion
```

La API de inferencia utiliza exclusivamente este modelo para responder predicciones.

## Challenger

Corresponde al modelo recién entrenado.

Antes de ser promovido, es comparado contra el Champion actual.

## Comparación

El sistema evalúa las métricas registradas para ambos modelos.

Actualmente la comparación se realiza principalmente utilizando:

```text
RMSE
```

donde un valor menor indica mejor desempeño.

## Regla de Promoción

El Challenger es promovido automáticamente cuando:

```text
RMSE Challenger < RMSE Champion
```

En ese caso:

1. Se actualiza el alias Champion en MLflow.
2. El Challenger se convierte en la nueva versión productiva.
3. La API detecta automáticamente el cambio.
4. El modelo es recargado sin intervención manual.

## Rechazo de Promoción

Si el Challenger no supera al Champion:

```text
RMSE Challenger >= RMSE Champion
```

el modelo permanece registrado en MLflow para trazabilidad, pero no es desplegado.

De esta forma se evita degradar el rendimiento del sistema en producción.


---

# Licencia

Este proyecto se distribuye bajo la licencia incluida en el archivo:

```text
LICENSE
```
 