# Entorno de Desarrollo con JupyterLab (Soporte Opcional para GPU)

Este directorio contiene la configuración necesaria para crear un entorno de desarrollo para ciencia de datos y machine learning utilizando JupyterLab, Docker y Docker Compose, con soporte opcional para GPUs NVIDIA.

## Descripción General

El objetivo de este proyecto es proporcionar un entorno de desarrollo reproducible y aislado. Opcionalmente, puede aprovechar la aceleración por hardware de las GPUs NVIDIA para el entrenamiento de modelos de inteligencia artificial.

El entorno base incluye:
- JupyterLab como IDE interactivo.
- Python 3.10.
- Librerías populares de ciencia de datos como TensorFlow, PyTorch, Pandas, y Scikit-learn.
- Soporte para CUDA y cuDNN a través de las imágenes base de NVIDIA (opcional, si se habilita el soporte para GPU).

## Prerrequisitos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

1.  **Docker y Docker Compose**: Para construir y gestionar los contenedores. [Instrucciones de instalación de Docker](https://docs.docker.com/get-docker/).

### Soporte Opcional para GPU (NVIDIA)

Si deseas utilizar la GPU de tu equipo, asegúrate de tener instalado lo siguiente:

1.  **NVIDIA GPU Drivers**: Drivers actualizados para tu tarjeta gráfica NVIDIA.
2.  **NVIDIA Container Toolkit**: Permite a Docker interactuar con las GPUs. [Instrucciones de instalación](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Cómo Usar

Sigue estos pasos para levantar el entorno de JupyterLab:

### 1. Construir e Iniciar el Contenedor

Abre una terminal en este directorio y ejecuta el siguiente comando:

```bash
docker compose up --build
```

-   `docker compose up`: Inicia los servicios definidos en el archivo `docker-compose.yml`.
-   `--build`: Fuerza la reconstrucción de la imagen de Docker. Deberías usar esta opción la primera vez que inicies el entorno o si has realizado cambios en el `Dockerfile` o en `requirements.txt`.

Para ejecutar el contenedor en segundo plano (detached mode), puedes usar:

```bash
docker compose up -d
```

### Habilitar Soporte para GPU (Opcional)

El archivo [`docker-compose.yml`](docker-compose.yml) ya tiene habilitado el soporte para GPU NVIDIA. Si no tienes una GPU NVIDIA o no deseas utilizarla, comenta las siguientes líneas en el archivo:

```yaml
# Descomentar en el caso de tener GPU nvidia en el equipo y desee usarla
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 2. Acceder a JupyterLab

Una vez que el contenedor esté en funcionamiento, abre tu navegador web y navega a la siguiente URL:

[http://localhost:8888](http://localhost:8888)

Verás la interfaz de JupyterLab, donde podrás crear, editar y ejecutar tus notebooks.

**Nota de Seguridad**: La configuración actual deshabilita la autenticación por token y contraseña para facilitar el desarrollo local. **No uses esta configuración en entornos de producción o accesibles públicamente**.

### 3. Detener el Contenedor

Para detener el entorno, puedes presionar `Ctrl + C` en la terminal donde se está ejecutando `docker compose up`.

Si el contenedor se está ejecutando en segundo plano, o desde una nueva terminal en el mismo directorio, ejecuta:

```bash
docker compose down
```

Este comando detendrá y eliminará el contenedor, pero tus archivos no se perderán gracias a los volúmenes que hemos configurado.

## Estructura del Proyecto

```
.
├── data/               # Datos de entrada para tus proyectos
├── Notebooks/          # Notebooks de Jupyter
├── results/            # Resultados de análisis y modelos
├── SRC/                # Código fuente Python
├── tests/              # Tests unitarios
├── Dockerfile          # Definición de la imagen Docker
├── docker-compose.yml  # Orquestación del contenedor
├── requirements.txt    # Dependencias de Python
└── README.md           # Esta documentación
```

### Archivos Principales

-   **[`Dockerfile`](Dockerfile)**: Contiene las instrucciones para construir la imagen de Docker. Define la imagen base de NVIDIA con CUDA 12.1.1 y cuDNN 8, instala Python 3.10 y las dependencias del sistema necesarias.
-   **[`docker-compose.yml`](docker-compose.yml)**: Orquesta la ejecución del contenedor. Define el servicio de JupyterLab, mapea el puerto 8888, configura el acceso a la GPU y gestiona los volúmenes para la persistencia de datos.
-   **[`requirements.txt`](requirements.txt)**: Lista todas las librerías de Python que se instalarán en el entorno, incluyendo TensorFlow, PyTorch, Pandas, Scikit-learn, y muchas otras herramientas para ciencia de datos.

## Persistencia de Datos

El archivo [`docker-compose.yml`](docker-compose.yml) está configurado para montar los siguientes directorios locales dentro del contenedor:

- `./data` → `/app/data`: Datos de entrada
- `./Notebooks` → `/app/Notebooks`: Notebooks de Jupyter
- `./results` → `/app/results`: Resultados y modelos
- `./SRC` → `/app/SRC`: Código fuente Python
- `./tests` → `/app/tests`: Tests unitarios

Cualquier archivo que crees o modifiques dentro de JupyterLab en estas rutas se guardará automáticamente en los directorios correspondientes de tu máquina local.

## Verificar el Uso de GPU

Si has habilitado el soporte para GPU, puedes verificar que TensorFlow y PyTorch detecten correctamente tu GPU ejecutando los siguientes comandos en una celda de notebook:

**Para TensorFlow:**
```python
import tensorflow as tf
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
```

**Para PyTorch:**
```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivos CUDA:", torch.cuda.device_count())
```

## Solución de Problemas

### El contenedor no inicia
- Verifica que Docker Desktop esté ejecutándose
- Asegúrate de que el puerto 8888 no esté ocupado por otra aplicación

### La GPU no es detectada
- Verifica que NVIDIA Container Toolkit esté instalado correctamente
- Ejecuta `nvidia-smi` en tu terminal para confirmar que los drivers funcionan
- Asegúrate de que las líneas de soporte GPU en [`docker-compose.yml`](docker-compose.yml) no estén comentadas

### Errores de memoria en GPU
- Reduce el tamaño del batch en tus modelos
- Cierra otros programas que puedan estar usando la GPU