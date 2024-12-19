# English: 
# Project Description

This project utilizes the OpenAI API and Azure Cognitive Search to manage documents and perform intelligent searches. Below are the key aspects of the repository:

## Key Features

1. **Loading Environment Variables**  
   - Variables are loaded from a `.env` file to configure the services.  
   - Includes validation to ensure that the required keys are correctly defined.  
   - Example variables: `OPENAI_API_KEY`, `SEARCH_SERVICE_ENDPOINT`, `SEARCH_API_KEY`, `INDEX_NAME`.

2. **Integration with OpenAI**  
   Configures OpenAI with support for models like `gpt-4` and `text-embedding-ada-002`. The configuration uses variables loaded from the `.env` file.

3. **Azure Cognitive Search Configuration**  
   - Enables the management of indexes and documents in Azure Search.  
   - Includes functions to validate the configuration and automatically create indexes if they don't exist.

4. **Document Upload to Azure Cognitive Search**  
   - Processes documents from a JSON file.  
   - Normalizes keys to comply with Azure Search requirements.  
   - Automates document uploads to the configured index.

5. **Relevant Document Search**  
   - Enables intelligent searches within the configured index.  
   - Returns the most relevant results based on a text query.

## Use Cases

- Managing documents in an Azure Cognitive Search index.  
- Implementing advanced searches using OpenAI language models.  
- Automating service configuration and data uploads from a defined environment.

This project is ideal for those seeking to integrate artificial intelligence and cloud services to manage and search large volumes of documents.

# Español:
# Descripción del Proyecto

Este proyecto utiliza la API de OpenAI y Azure Cognitive Search para gestionar documentos y realizar búsquedas inteligentes. A continuación, se describen los aspectos más importantes del repositorio:

## Características Principales

1. **Cargar Variables de Entorno**  
   - Las variables se cargan desde un archivo `.env` para configurar los servicios.
   - Incluye una validación para asegurarse de que las claves necesarias están correctamente definidas.
   - Ejemplo de variables: `OPENAI_API_KEY`, `SEARCH_SERVICE_ENDPOINT`, `SEARCH_API_KEY`, `INDEX_NAME`.

2. **Integración con OpenAI**  
   Configura OpenAI con soporte para modelos como `gpt-4` y `text-embedding-ada-002`. La configuración utiliza las variables cargadas desde el archivo `.env`.

3. **Configuración de Azure Cognitive Search**  
   - Permite gestionar índices y documentos en Azure Search.  
   - Incluye funciones para validar la configuración y crear índices automáticamente si no existen.

4. **Carga de Documentos en Azure Cognitive Search**  
   - Procesa documentos desde un archivo JSON.
   - Normaliza las claves para cumplir con los requisitos de Azure Search.
   - Subida automatizada al índice configurado.

5. **Búsqueda de Documentos Relevantes**  
   - Permite realizar búsquedas inteligentes en el índice configurado.
   - Devuelve los resultados más relevantes basados en una consulta de texto.

## Casos de Uso

- Gestión de documentos en un índice de Azure Cognitive Search.
- Implementación de búsquedas avanzadas utilizando modelos de lenguaje de OpenAI.
- Automatización de la configuración de servicios y carga de datos desde un entorno definido.

Este proyecto es ideal para quienes buscan integrar inteligencia artificial y servicios en la nube para gestionar y buscar grandes volúmenes de documentos.
