FROM apache/airflow:2.9.1-python3.11

USER airflow

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

# Actualizar pip para evitar warnings
RUN pip install --upgrade pip

# Copiar requirements
COPY requirements-airflow.txt /requirements.txt

# Instalar dependencias (sin --no-cache para debug si falla)
RUN pip install -r /requirements.txt

# Copiar DAGs
COPY airflow/dags /opt/airflow/dags

# Copiar código fuente
COPY airflow/src /opt/airflow/src

# Copiar plugins
COPY airflow/plugins /opt/airflow/plugins

# Agregar carpeta .git
COPY .git /opt/airflow/.git

# Asegurar imports desde /opt/airflow
ENV PYTHONPATH=/opt/airflow