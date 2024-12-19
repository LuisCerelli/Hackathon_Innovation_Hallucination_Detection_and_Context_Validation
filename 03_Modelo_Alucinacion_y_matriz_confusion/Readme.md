# English: 
# Factuality Validation Project

This project implements a model that evaluates the factuality of generated responses, using a combination of Azure Cognitive Search tools, OpenAI, and machine learning models. Here you will find instructions to set up the environment and run the program.

## Prerequisites

- Python 3.8 or higher.
- Azure account with access to Azure Cognitive Search.
- API key and access to OpenAI services.

## Installation

### 1. Configure the Correct OpenAI Version
This project requires a specific version of the `openai` library. Ensure you install the correct version:

```bash
pip uninstall openai
pip install openai==0.27.8
```

### 2. Install Required Libraries
Install the necessary dependencies by running the following command:

```bash
pip install openai azure-search-documents python-dotenv transformers scikit-learn matplotlib seaborn numpy spacy tiktoken
python -m spacy download es_core_news_sm
```

## Configuration

### Environment Variables

Create a `.env` file in the project's root directory with the following environment variables:

```env
# Azure Cognitive Search
SEARCH_SERVICE_ENDPOINT=<YOUR_AZURE_SEARCH_ENDPOINT>
SEARCH_API_KEY=<YOUR_AZURE_SEARCH_API_KEY>
INDEX_NAME=<INDEX_NAME>

# OpenAI
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
MODEL_NAME=text-embedding-ada-002

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=<YOUR_AZURE_OPENAI_ENDPOINT>
AZURE_OPENAI_KEY=<YOUR_AZURE_OPENAI_API_KEY>
ENGINE_NAME=<ENGINE_NAME>
```

### `.env` File Verification
The program automatically searches for the `.env` file and displays a message indicating if it is found correctly.

## Usage

### Main Workflow

The main workflow evaluates whether a response is factual and generates a confusion matrix to analyze performance.

### Key Features

1. **Embedding Generation**:
    - Splits text into fragments and computes embeddings using OpenAI.
    - Validates that embeddings do not contain invalid values.

2. **Factuality Validation**:
    - Uses a pipeline based on the `mrm8488/bert-multi-cased-finetuned-xquadv1` model to analyze the response.
    - Compares embeddings to calculate semantic similarity.

3. **Azure Search**:
    - Searches for relevant reference documents in Azure Cognitive Search.

4. **Confusion Matrix**:
    - Generates and visualizes the confusion matrix to evaluate the model's performance.

## Visualization

The program generates a confusion matrix and a detailed classification report.
# Español
# Proyecto de Validación de Factualidad

Este proyecto implementa un modelo que evalúa la factualidad de respuestas generadas, utilizando una combinación de herramientas de Azure Cognitive Search, OpenAI y modelos de machine learning. Aquí encontrarás instrucciones para configurar el entorno y ejecutar el programa.

## Requisitos Previos

- Python 3.8 o superior.
- Cuenta de Azure con acceso a Azure Cognitive Search.
- Clave de API y acceso a los servicios de OpenAI.

## Instalación

### 1. Configurar la Versión Correcta de OpenAI
Este proyecto requiere una versión específica de la librería `openai`. Asegúrate de instalar la versión correcta:

```bash
pip uninstall openai
pip install openai==0.27.8
```

### 2. Instalar Librerías Necesarias
Instala las dependencias requeridas ejecutando el siguiente comando:

```bash
pip install openai azure-search-documents python-dotenv transformers scikit-learn matplotlib seaborn numpy spacy tiktoken
python -m spacy download es_core_news_sm
```

## Configuración

### Variables de Entorno

Crea un archivo `.env` en el directorio raíz del proyecto con las siguientes variables de entorno:

```env
# Azure Cognitive Search
SEARCH_SERVICE_ENDPOINT=<TU_ENDPOINT_AZURE_SEARCH>
SEARCH_API_KEY=<TU_CLAVE_API_AZURE_SEARCH>
INDEX_NAME=<NOMBRE_DEL_INDICE>

# OpenAI
OPENAI_API_KEY=<TU_CLAVE_API_OPENAI>
MODEL_NAME=text-embedding-ada-002

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=<TU_ENDPOINT_AZURE_OPENAI>
AZURE_OPENAI_KEY=<TU_CLAVE_API_AZURE_OPENAI>
ENGINE_NAME=<NOMBRE_DEL_ENGINE>
```

### Verificación del Archivo `.env`
El programa busca automáticamente el archivo `.env` y muestra un mensaje indicando si se encuentra correctamente.

## Uso

### Flujo Principal

El flujo principal evalúa si una respuesta es factual y genera una matriz de confusión para analizar el rendimiento.

### Funcionalidades Principales

1. **Generación de Embeddings**:
    - Divide texto en fragmentos y calcula embeddings utilizando OpenAI.
    - Valida que los embeddings no contengan valores inválidos.

2. **Validación de Factualidad**:
    - Utiliza un pipeline basado en el modelo `mrm8488/bert-multi-cased-finetuned-xquadv1` para analizar la respuesta.
    - Compara embeddings para calcular similitud semántica.

3. **Búsqueda en Azure**:
    - Busca documentos de referencia relevantes en Azure Cognitive Search.

4. **Matriz de Confusión**:
    - Genera y visualiza la matriz de confusión para evaluar el desempeño del modelo.

## Visualización

El programa genera una matriz de confusión y un reporte de clasificación detallado.


