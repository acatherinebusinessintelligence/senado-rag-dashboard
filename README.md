# 🧠 Senado RAG Dashboard

Sistema interactivo para el análisis de proyectos de ley y actividad legislativa en Colombia usando Inteligencia Artificial.

## 🚀 ¿Qué hace este proyecto?

Este dashboard permite:

- 🔎 Búsqueda semántica sobre proyectos de ley (RAG)
- 🏛️ Análisis de participación de senadores
- 📊 Visualización de afinidad temática
- 🔗 Relación entre proyectos de ley y plenarias
- 🧠 Exploración de información legislativa con IA

## 🏗️ Arquitectura

- **RAG (Retrieval-Augmented Generation)** con ChromaDB
- **Embeddings** con Sentence Transformers
- **Visualización** con Plotly
- **Interfaz** con Streamlit
- **Procesamiento de datos** con Pandas y NLP

## 🧩 Componentes

- Corpus de proyectos de ley
- Corpus de plenarias
- Índice vectorial (Chroma)
- Roles enriquecidos de senadores
- Fotos y metadatos

## ⚠️ Nota sobre datos

Este repositorio no incluye:

- `artifacts/`
- `data/`
- `fotos/`

Debido a su tamaño y naturaleza.

## 🛠️ Instalación

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

