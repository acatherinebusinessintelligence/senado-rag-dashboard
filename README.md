# 🏛️ Senado RAG Dashboard

Sistema de análisis legislativo basado en **RAG (Retrieval-Augmented Generation)**, que permite explorar proyectos de ley, plenarias y comportamiento de senadores mediante búsqueda semántica, evidencia documental y analítica visual.

---

## 🌐 Demo visual del proyecto

👉 **Explorar la landing del sistema**  
https://acatherinebusinessintelligence.github.io/senado-rag-dashboard/#vision

---

## 🎯 ¿Qué resuelve?

La información legislativa suele ser:

- dispersa  
- extensa  
- difícil de interpretar rápidamente  

Este proyecto transforma esos datos en una **experiencia navegable y accionable**, permitiendo:

- entender proyectos de ley en contexto  
- cruzar información con plenarias  
- analizar comportamiento legislativo  
- explorar relaciones entre actores políticos  

---

## 🔍 Capacidades principales

### 1. Búsqueda semántica (RAG)
- Consultas en lenguaje natural  
- Recuperación de evidencia relevante  
- Uso de embeddings + ChromaDB  

### 2. Visual analytics
- Métricas de relevancia  
- Gráficos interpretables  
- Rankings y tendencias  

### 3. Exploración de senadores
- Perfil enriquecido  
- Relación con proyectos de ley  
- Mapas y grafos temáticos  

---

## 🧠 Arquitectura

```text
Ingesta de datos → Normalización → Embeddings → ChromaDB → Dashboard (Streamlit)
