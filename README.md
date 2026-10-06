# Sistema de Recuperación Semántica de Imágenes (CBIR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tu-usuario/tu-repo/blob/main/notebooks/Sistema_Recuperacion_Semantica_Imagenes.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/tu-usuario/tu-repo/blob/main/notebooks/Sistema_Recuperacion_Semantica_Imagenes.ipynb)

## Resumen del Proyecto

Este repositorio contiene la implementación de un sistema avanzado de **Recuperación de Imágenes Basada en Contenido (CBIR)**, apoyado en modelos de visión-lenguaje y recuperación vectorial.

### Problema Analítico
El desafío principal es recuperar imágenes de un gran conjunto de datos (Pascal VOC 2012) utilizando consultas en lenguaje natural complejo que pueden incluir **negaciones y exclusiones** (ej. "una calle sin coches"). Los sistemas CBIR tradicionales fallan al entender la semántica profunda de las exclusiones. Nuestro objetivo es optimizar las consultas del usuario y penalizar los conceptos excluidos.

### Pipeline de Datos y Arquitectura
El pipeline se divide en tres etapas principales:
1. **Representación Vectorial**: Uso de **CLIP (ViT-B/32)** para generar embeddings semánticos densos de las imágenes y consultas.
2. **Base de Datos Vectorial**: Indexación de los embeddings de imágenes en memoria utilizando **FAISS (IndexFlatIP)** para una búsqueda de similitud ultrarrápida (búsqueda de producto interno / similitud de coseno).
3. **Reformulación y Reranking con LLM**: Integración de un Large Language Model cuantizado a 4-bits (**Phi-3-mini-4k-instruct**) para traducir, enriquecer con sinónimos y extraer conceptos negados de las consultas del usuario, aplicando posteriormente un mecanismo de *Reranking* para penalizar matemáticamente los elementos excluidos.

### Métricas y Hallazgos
- **Clustering Semántico**: El análisis con K-Means y UMAP demuestra una fuerte cohesión en los embeddings de CLIP a lo largo de las distintas clases de objetos.
- **Rendimiento de Recuperación**: El módulo de Reranking penalizado incrementa significativamente la precisión (top-K precision) cuando los usuarios introducen negaciones complejas, comparado con un *baseline* estricto de CLIP.

## Estructura del Repositorio

```plaintext
├── README.md                 # Caso de estudio: problema, metodología y conclusiones
├── requirements.txt          # Dependencias fijadas
├── data/
│   ├── raw/                  # Datos crudos (Imágenes Pascal VOC - ver instrucciones)
│   └── processed/            # Matrices FAISS y Embeddings precalculados
├── src/
│   └── utils.py              # Funciones auxiliares de preprocesamiento (opcional)
└── notebooks/
    └── Sistema_Recuperacion_Semantica_Imagenes.ipynb  # Notebook principal (Fully Executed)
```

## Reproducibilidad

Todo el código está contenido en `notebooks/Sistema_Recuperacion_Semantica_Imagenes.ipynb`, pre-ejecutado para su revisión sin necesidad de correr un servidor local. 

**Nota sobre los Datos**: El dataset utilizado es **Pascal VOC 2012**. Debido a su tamaño, no está incluido en este repositorio. El notebook descarga o asume la existencia de los datos a través de Kaggle Datasets de forma dinámica y utiliza rutas relativas, previniendo errores de entorno.
