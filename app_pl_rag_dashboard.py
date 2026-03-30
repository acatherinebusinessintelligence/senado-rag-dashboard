#-*- coding: utf-8 -*-
"""
app_pl_rag_dashboard.py
FIX + UI cálida (mobile-first) + senadores desde roles_enriched + cero fricción

✅ Corrige:
- ValueError: truth value of an empty array is ambiguous (pd.notna sobre listas/arrays)
- StreamlitDuplicateElementId (plotly_chart con key único)
- Tabs mal indexados
- attach_senators_from_roles (indentación/llaves)
- Duplicados de funciones (este archivo deja UNA sola versión de cada helper)

✅ UI estilo confirmado:
- Mobile-first, cards grandes, bordes suaves, espacios amplios
- Control principal: “¿Qué te preocupa hoy?” + botón “Explícame fácil”
- Chips, tooltips, panel lateral, expanders para evidencia

Requisitos:
pip install streamlit pandas numpy plotly chromadb sentence-transformers
Opcional:
pip install networkx psutil

Ejecución:
streamlit run app_pl_rag_dashboard.py
"""

# ======================================================================
# [BLOCK 01] Imports + Config
# ======================================================================
import os
import sys
import json
import math
import re
import time
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import streamlit as st

def decide_scope(query: str, selected: str) -> str:
    """Decide el scope final. selected: Auto/Proyectos de Ley/Plenarias/Ambos."""
    if selected and selected != "Auto":
        return selected
    q = (query or "").lower()
    pl_kw = ["pl", "proyecto de ley", "artículo", "articulo", "texto", "ponencia", "gaceta", "articulado", "exposición de motivos", "exposicion de motivos"]
    plen_kw = ["senador", "senadora", "dijo", "intervino", "plenaria", "debate", "sesión", "sesion", "votación", "votacion"]
    pl_hit = any(k in q for k in pl_kw)
    plen_hit = any(k in q for k in plen_kw)
    if pl_hit and not plen_hit:
        return "Proyectos de Ley"
    if plen_hit and not pl_hit:
        return "Plenarias"
    if pl_hit and plen_hit:
        return "Ambos"
    return "Ambos"

import plotly.express as px
import plotly.graph_objects as go

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import psutil
except Exception:
    psutil = None


APP_TITLE = "Senado RAG — Proyectos de Ley (PL) | Dashboard"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR = Path(__file__).resolve().parent

def _prefer_existing(*candidates: str) -> str:
    """
    Devuelve la primera ruta existente. Si ninguna existe, devuelve la primera.
    Permite mantener compatibilidad local sin amarrar la app a una sola ruta.
    """
    cleaned = [str(c) for c in candidates if c]
    for c in cleaned:
        if Path(c).exists():
            return c
    return cleaned[0] if cleaned else ""

DEFAULT_CORPUS = _prefer_existing(
    str(BASE_DIR / "artifacts" / "pl_rag_corpus.parquet"),
    "artifacts/pl_rag_corpus.parquet",
)
DEFAULT_CHROMA_DIR = _prefer_existing(
    str(BASE_DIR / "artifacts" / "chroma_pl"),
    "artifacts/chroma_pl",
)
DEFAULT_COLLECTION = "pl_chunks"

PL_TEXT_CANON = _prefer_existing(
    str(BASE_DIR / "artifacts" / "pl_text_canon_from_folders.parquet"),
    "artifacts/pl_text_canon_from_folders.parquet",
)
ROLES_SUMMARY = _prefer_existing(
    str(BASE_DIR / "data" / "PL_SENADORES" / "pl_roles_summary.parquet"),
    "data/PL_SENADORES/pl_roles_summary.parquet",
)
ROLES_ENRICHED = _prefer_existing(
    str(BASE_DIR / "data" / "PL_SENADORES" / "pl_senadores_roles_enriched.parquet"),
    "data/PL_SENADORES/pl_senadores_roles_enriched.parquet",
)

DEFAULT_PLENARIAS_CORPUS = _prefer_existing(
    str(BASE_DIR / "artifacts" / "plenarias_rag_corpus.parquet"),
    "artifacts/plenarias_rag_corpus.parquet",
)
DEFAULT_PLENARIAS_COLLECTION = "plenaria_chunks"
DEFAULT_PLENARIAS_CHROMA_DIR = DEFAULT_CHROMA_DIR

DEFAULT_PLENARIAS_DIR = _prefer_existing(
    str(BASE_DIR / "data" / "Plenarias"),
    r"C:\Users\Demo\Documents\Proyectos\Elecciones\Senado\Codigo\senado_rag_streamlit_app_v4_candidates_ocr\senado_rag_streamlit_app_v4_candidates_ocr\data\Plenarias",
)
DEFAULT_SENADORES_CSV = _prefer_existing(
    str(BASE_DIR / "data" / "HV_SENADORES" / "senadores.csv"),
    r"C:\Users\Demo\Documents\Proyectos\Elecciones\Senado\Codigo\senado_rag_streamlit_app_v4_candidates_ocr\senado_rag_streamlit_app_v4_candidates_ocr\data\HV_SENADORES\senadores.csv",
    "data/HV_SENADORES/senadores.csv",
)
DEFAULT_SENADORES_FOTOS_DIR = _prefer_existing(
    str(BASE_DIR / "fotos"),
    "fotos",
)
SENADORES_CSV = DEFAULT_SENADORES_CSV
YOUTUBE_BASE = "https://www.youtube.com/watch?v="


SENADO_BOILERPLATE_PATTERNS = [
    r"Enviar\s+Mensaje\s+Enviar\s+Volver\s+a\s+Senadores",
    r"Capitolio\s+Nacional\.?\s+Plaza\s+de\s+Bolívar",
    r"PBX\s+General:\s*\(?57\)?",
    r"Atención\s+Ciudadana\s+del\s+Congreso",
    r"Formulario\s+electrónico\s+derechos\s+de\s+petición",
    r"Notificaciones\s+Judiciales",
    r"Senado\s+de\s+la\s+República\s+de\s+Colombia",
    r"Comisiones\s+Constitucionales\s+Permanentes",
    r"Transparencia\s+y\s+Acceso\s+a\s+la\s+Información",
    r"Formulario\s+PQRSD",
    r"Participación\s+Ciudadana",
]


# ======================================================================
# [BLOCK 02] State + Estilos UI (cálido, sobrio)
# ======================================================================
def init_state():
    defaults = {
        "_chart_i": 0,
        "_show_log": False,
        "_last_cmd": "",
        "_last_out": "",
        "_last_rc": None,
        "hits_df": None,
        "hits_dbg": None,
        "mode_expert": False,
        "theme_selected": "(todos)",
        "senator_source": "Desde roles_enriched.parquet (si existe)",
        "ux_explain": True,
        "main_worry": "glifosato",
        "k": 8,
        "fetch_k": 80,
        "filter_bad_ocr": True,
        "filter_bad_ocr_on": True,
        "lexical_rerank_on": True,
        "pl_norm_filter": "",
        "section_filter": "(todas)",
        "doc_kind_filter": "(todos)",
        "show_text_chars": 1200,
        "evidence_source": "Ambas (PL + Plenaria)",
        "plenarias_corpus_path": DEFAULT_PLENARIAS_CORPUS,
        "senadores_csv_path": DEFAULT_SENADORES_CSV,
        "fotos_dir_path": DEFAULT_SENADORES_FOTOS_DIR,
        "plenarias_chroma_dir": DEFAULT_PLENARIAS_CHROMA_DIR,
        "plenarias_collection": DEFAULT_PLENARIAS_COLLECTION,
        "search_scope": "Auto",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


st.set_page_config(page_title=APP_TITLE, layout="wide")

init_state()

WARM_CSS = """
<style>
/* Mobile-first feel */
.block-container {padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1150px;}
h1, h2, h3 {letter-spacing: -0.02em;}
/* Softer cards */
div[data-testid="stContainer"] {border-radius: 18px;}
/* Inputs */
div[data-baseweb="input"] input {border-radius: 14px;}
div[data-baseweb="select"] > div {border-radius: 14px;}
/* Buttons */
button[kind="primary"] {border-radius: 14px;}
/* Chips look for radio/segmented */
div[role="radiogroup"] {gap: 0.35rem;}
</style>
"""
st.markdown(WARM_CSS, unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("Paso a paso, cero fricción. Primero RAG de PL; luego (si quieres) plenarias.")

# ======================================================================
# [BLOCK 03] Utils: lectura, texto, metadatos, plotly keys
# ======================================================================
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def tokenize(s: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(s or "")]

def lexical_score(query: str, doc: str) -> float:
    q = tokenize(query)
    d = tokenize(doc)
    if not q or not d:
        return 0.0
    qset, dset = set(q), set(d)
    inter = len(qset & dset)
    return inter / math.sqrt(len(qset) * len(dset) + 1e-9)

def looks_like_bad_ocr(text: str) -> bool:
    if not text:
        return True
    t = str(text).strip()
    if len(t) < 80:
        return True
    letters = sum(c.isalpha() for c in t)
    if letters / max(len(t), 1) < 0.55:
        return True
    weird = len(re.findall(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ\.,;:\-\(\)\[\]\"'/%]", t))
    if weird / max(len(t), 1) > 0.08:
        return True
    return False

def safe_read_parquet(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        st.error(f"No pude leer parquet: {path}\n{e}")
        return None


@st.cache_data(show_spinner=False)

def remove_senado_boilerplate(text: str) -> str:
    if not text:
        return text
    t = text
    for pat in SENADO_BOILERPLATE_PATTERNS:
        t = re.sub(pat + r".*$", "", t, flags=re.IGNORECASE | re.DOTALL)
    return t.strip()


def youtube_url_from_meta(row: pd.Series) -> Optional[str]:
    """
    Construye link YouTube por chunk si hay m_video_id y (opcional) m_start_sec.
    Soporta llaves m_video_id/video_id y m_start_sec/start_sec.
    """
    vid = row.get("m_video_id", None) or row.get("video_id", None)
    if not vid:
        return None
    vid = str(vid).strip()
    if not vid:
        return None

    start = row.get("m_start_sec", None) or row.get("start_sec", None) or 0
    try:
        start_i = int(float(start))
    except Exception:
        start_i = 0

    if start_i > 0:
        return f"{YOUTUBE_BASE}{vid}&t={start_i}s"
    return f"{YOUTUBE_BASE}{vid}"


def get_senator_profile_url(name: str, sen_df: Optional[pd.DataFrame]) -> Optional[str]:
    """Devuelve la URL de perfil del senador desde senadores.csv (tolerante a esquemas).

    Espera idealmente columnas:
      - name_norm (nombre normalizado)
      - perfil_url (url ficha)
    Pero soporta variantes comunes (nombre, name, senador_nombre, url, link, perfil, etc.).
    """
    if sen_df is None or sen_df.empty or not name:
        return None

    def norm_name(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-záéíóúüñ0-9 ]", "", s)
        return s

    key = norm_name(name)

    # --- asegurar columnas (no asumimos esquema fijo del CSV) ---
    df = sen_df

    # name_norm
    if "name_norm" not in df.columns:
        # intenta derivarla de una columna de nombre existente
        name_cols = [c for c in ["name", "nombre", "senador_nombre", "nombre_completo", "senador", "Senador"] if c in df.columns]
        if name_cols:
            src = name_cols[0]
            try:
                df = df.copy()
                df["name_norm"] = df[src].fillna("").astype(str).map(norm_name)
            except Exception:
                # si algo raro pasa, al menos evita KeyError
                df = df.copy()
                df["name_norm"] = ""
        else:
            df = df.copy()
            df["name_norm"] = ""

    # perfil_url (acepta alias)
    url_col = None
    if "perfil_url" in df.columns:
        url_col = "perfil_url"
    else:
        for c in ["url", "link", "perfil", "profile_url", "href", "ficha_url"]:
            if c in df.columns:
                url_col = c
                break

    if url_col is None:
        return None

    # match exacto por name_norm
    sub = df[df["name_norm"] == key]
    if not sub.empty:
        u = str(sub.iloc[0].get(url_col, "")).strip()
        return u or None

    # fallback: contains (cuando el nombre venga con doble apellido/cambios)
    try:
        sub2 = df[df["name_norm"].astype(str).str.contains(re.escape(key), na=False)]
    except Exception:
        sub2 = df[df["name_norm"].apply(lambda x: key in str(x))]
    if not sub2.empty:
        u = str(sub2.iloc[0].get(url_col, "")).strip()
        return u or None

    return None

def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out
    except Exception as e:
        return 999, str(e)

def ensure_plotly_fig_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def _next_chart_key(prefix: str = "chart") -> str:
    st.session_state["_chart_i"] += 1
    return f"{prefix}_{st.session_state['_chart_i']}"

def _next_btn_key(prefix: str = "btn") -> str:
    st.session_state.setdefault("_btn_i", 0)
    st.session_state["_btn_i"] += 1
    return f"{prefix}_{st.session_state['_btn_i']}"


def show_fig(fig: go.Figure, height: int = 360, key_prefix: str = "chart"):
    st.plotly_chart(
        ensure_plotly_fig_layout(fig, height), use_container_width=True,
        key=_next_chart_key(key_prefix),
    )

def coerce_meta_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return s
        return s
    return v


def explode_listlike(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    tmp = df.copy()

    def to_list(x):
        x = coerce_meta_value(x)
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        return obj
                    return [str(obj)]
                except Exception:
                    return [s]
            return [s]
        return [str(x)]


    tmp[col] = tmp[col].apply(to_list)
    tmp = tmp.explode(col)
    tmp[col] = tmp[col].fillna("").astype(str)
    tmp = tmp[tmp[col].str.strip() != ""]
    return tmp

# ----------------------------
# Plenarias: YouTube links + Senadores directory links
# ----------------------------
def youtube_url(video_id: Any, start_sec: Any = 0) -> Optional[str]:
    vid = str(video_id or "").strip()
    if not vid or vid.lower() in ("nan", "none", "na"):
        return None
    ss = 0
    try:
        ss = int(float(start_sec or 0))
    except Exception:
        ss = 0
    ss = max(ss, 0)
    return f"https://www.youtube.com/watch?v={vid}&t={ss}s"

@st.cache_data(show_spinner=False)
def load_senadores_directory(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=["senador_nombre", "perfil_url"])
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        df = pd.read_csv(p, encoding="latin-1")

    # Normaliza nombres de columnas esperadas
    ren = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("senador", "nombre", "senador_nombre", "nombre_senador", "name"):
            ren[c] = "senador_nombre"
        elif cl in ("perfil_url", "url", "link", "perfil", "hoja_vida_url"):
            ren[c] = "perfil_url"
    df = df.rename(columns=ren)

    if "senador_nombre" not in df.columns:
        return pd.DataFrame(columns=["senador_nombre", "perfil_url"])

    if "perfil_url" not in df.columns:
        df["perfil_url"] = ""

    df["senador_nombre"] = df["senador_nombre"].fillna("").astype(str).str.strip()
    df["perfil_url"] = df["perfil_url"].fillna("").astype(str).str.strip()

    df = df[(df["senador_nombre"] != "")].copy()
    df = df.drop_duplicates(subset=["senador_nombre"], keep="first")
    return df[["senador_nombre", "perfil_url"]].reset_index(drop=True)

def render_senadores_links(names: List[str], sen_dir: Optional[pd.DataFrame]) -> str:
    if not names:
        return "—"
    if sen_dir is None or sen_dir.empty:
        return ", ".join(names)

    lookup = {r["senador_nombre"]: r["perfil_url"] for _, r in sen_dir.iterrows() if str(r.get("perfil_url","")).strip()}
    parts = []
    for nm in names:
        url = lookup.get(nm, "")
        if url:
            parts.append(f"[{nm}]({url})")
        else:
            parts.append(nm)
    return ", ".join(parts)


# ============================================================
# UI HELPERS (Tab 1) - "explicame fácil" + cards + métricas
# ============================================================

def is_present_value(v: Any) -> bool:
    """Evita pd.notna() con listas/arrays (causa ValueError truth value...)."""
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    if isinstance(v, str):
        s = v.strip()
        return s != "" and s.lower() not in ("nan", "none", "null", "na", "—")

    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0

    try:
        if isinstance(v, np.ndarray):
            return v.size > 0
    except Exception:
        pass

    return True


def list_to_text(v: Any, max_items: int = 8) -> str:
    """Convierte listas/strings/jsonlist a texto bonito."""
    arr = _to_list_str(v)
    arr = [x for x in arr if x and str(x).strip()]
    if not arr:
        return "—"
    arr = arr[:max_items]
    return ", ".join(arr)


def pick_best_evidence(sub: pd.DataFrame) -> pd.Series:
    """Selecciona el chunk 'mejor' para mostrar (objeto/exposición primero)."""
    if sub is None or sub.empty:
        return None
    preferred_sections = ["objeto", "exposicion_motivos", "cuerpo", "articulado"]
    for s in preferred_sections:
        cand = sub[sub.get("m_section", "").astype(str) == s]
        if len(cand) > 0:
            return cand.sort_values(["distance", "lex_score"], ascending=[True, False]).iloc[0]
    return sub.sort_values(["distance", "lex_score"], ascending=[True, False]).iloc[0]


def confidence_label(min_dist: float) -> str:
    # Ajusta a tu distribución real; tus distancias están ~1.04-1.18
    if min_dist <= 1.06:
        return "Alta"
    if min_dist <= 1.12:
        return "Media"
    return "Baja"

def normalize_corpus_columns(df_corpus: pd.DataFrame) -> pd.DataFrame:
    df = df_corpus.copy()

    # meta -> m_*
    if "meta" in df.columns:
        def load_meta(x):
            if x is None:
                return {}
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return {}
                try:
                    return json.loads(s)
                except Exception:
                    return {"meta_raw": s}
            return {"meta_raw": str(x)}

        metas = df["meta"].apply(load_meta)
        meta_df = pd.json_normalize(metas)
        meta_df.columns = [f"m_{c}" for c in meta_df.columns]
        df = pd.concat([df.drop(columns=["meta"]), meta_df], axis=1)

    # copia base -> m_*
    base_cols = [
        "pl_norm", "section", "doc_kind", "doc_file",
        "comisiones", "partidos_unicos", "senadores", "temas",
        "titulo", "objeto"
    ]
    for base in base_cols:
        if base in df.columns and f"m_{base}" not in df.columns:
            df[f"m_{base}"] = df[base]

    for c in ["m_pl_norm", "m_section", "m_doc_kind", "m_doc_file"]:
        if c in df.columns:
            df[c] = df[c].fillna("NA").astype(str)

    return df

@st.cache_data(show_spinner=False)
def get_top_themes_from_corpus(df_norm: Optional[pd.DataFrame], top_n: int = 60) -> List[str]:
    if df_norm is None or df_norm.empty:
        return []
    if "m_temas" not in df_norm.columns:
        return []
    tmp = explode_listlike(df_norm[["m_temas"]].copy(), "m_temas")
    if tmp.empty:
        return []
    return tmp["m_temas"].value_counts().head(top_n).index.astype(str).tolist()

def filter_hits_by_theme(hits_df: pd.DataFrame, theme_selected: str) -> pd.DataFrame:
    if hits_df is None or hits_df.empty:
        return hits_df
    if theme_selected in (None, "", "(todos)"):
        return hits_df
    if "m_temas" not in hits_df.columns:
        return hits_df

    tmp = hits_df.copy()

    def has_theme(x):
        vals = coerce_meta_value(x)
        if vals is None:
            return False
        if isinstance(vals, list):
            return theme_selected in [str(v) for v in vals]
        if isinstance(vals, str):
            s = vals.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        return theme_selected in [str(v) for v in obj]
                except Exception:
                    pass
            return theme_selected.lower() in s.lower()
        return False

    mask = tmp["m_temas"].apply(has_theme)
    return tmp[mask].reset_index(drop=True)


# ======================================================================
# [BLOCK 04] Roles enriched + attach senadores
# ======================================================================
@st.cache_data(show_spinner=False)
def safe_read_roles_enriched(path: str) -> Optional[pd.DataFrame]:
    df = safe_read_parquet(path)
    if df is None or df.empty:
        return None

    ren = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("senador", "senador_nombre", "senador_name", "nombre_senador"):
            ren[c] = "senador_nombre"
        elif cl in ("pl", "pl_norm", "plnorm", "plid"):
            ren[c] = "pl_norm"
        elif cl in ("chunk_id", "chunkid", "id"):
            ren[c] = "chunk_id"
        elif cl in ("rol", "role"):
            ren[c] = "rol"
        elif cl in ("score", "peso", "weight", "puntaje"):
            ren[c] = "score"
        elif cl in ("section", "seccion"):
            ren[c] = "section"
        elif cl in ("doc_kind", "dockind"):
            ren[c] = "doc_kind"
        elif cl in ("doc_file", "docfile", "file"):
            ren[c] = "doc_file"
        elif cl in ("partido", "party"):
            ren[c] = "partido"
        elif cl in ("comision", "comisiones", "commission"):
            ren[c] = "comision"
    df = df.rename(columns=ren)

    # mínimos
    if "senador_nombre" not in df.columns or "pl_norm" not in df.columns:
        return None

    if "rol" not in df.columns:
        df["rol"] = "participa"
    if "score" not in df.columns:
        df["score"] = 1.0

    df["senador_nombre"] = df["senador_nombre"].fillna("").astype(str)
    df["pl_norm"] = df["pl_norm"].fillna("").astype(str)
    df["rol"] = df["rol"].fillna("participa").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1.0)

    if "chunk_id" in df.columns:
        df["chunk_id_int"] = pd.to_numeric(df["chunk_id"], errors="coerce").astype("Int64")
    else:
        df["chunk_id_int"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # llaves opcionales (por si quieres fallback exacto)
    for c in ["section", "doc_kind", "doc_file"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df

def _extract_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _to_list_str(x: Any) -> List[str]:
    x = coerce_meta_value(x)
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass
    return [s]

def attach_senators_from_roles(
    hits_df: pd.DataFrame,
    roles_df: Optional[pd.DataFrame],
    prefer_chunk_id: bool = True,
    out_col: str = "m_senadores",
    keep_detail_col: str = "m_senadores_roles_detail",
    max_names: int = 10,
) -> pd.DataFrame:
    """
    Enriquecer hits con senadores desde roles_enriched.
    1) join por chunk_id (id del hit) si posible
    2) fallback por (m_pl_norm, m_section, m_doc_kind, m_doc_file) si existen
    """
    if hits_df is None or hits_df.empty:
        return hits_df
    if roles_df is None or roles_df.empty:
        return hits_df

    h = hits_df.copy()
    r = roles_df.copy()

    # prepara chunk_id_int en hits
    h["chunk_id_int"] = h["id"].apply(_extract_int).astype("Int64")

    merged = h.copy()

    # 1) JOIN chunk_id
    if prefer_chunk_id and r["chunk_id_int"].notna().any() and merged["chunk_id_int"].notna().any():
        merged = merged.merge(
            r[["chunk_id_int", "senador_nombre", "rol", "score", "pl_norm"] +
              ([c for c in ["section", "doc_kind", "doc_file"] if c in r.columns])],
            on="chunk_id_int",
            how="left",
            suffixes=("", "_r"),
        )

    # 2) Fallback por llaves si falta senador y tenemos llaves
    has_hit_keys = all(c in merged.columns for c in ["m_pl_norm", "m_section", "m_doc_kind", "m_doc_file"])
    has_role_keys = all(c in r.columns for c in ["pl_norm", "section", "doc_kind", "doc_file"])

    if has_hit_keys and has_role_keys:
        need = merged["senador_nombre"].isna() if "senador_nombre" in merged.columns else pd.Series([True]*len(merged))
        if need.any():
            fb = merged[need].copy()
            fb = fb.merge(
                r[["pl_norm", "section", "doc_kind", "doc_file", "senador_nombre", "rol", "score"]],
                left_on=["m_pl_norm", "m_section", "m_doc_kind", "m_doc_file"],
                right_on=["pl_norm", "section", "doc_kind", "doc_file"],
                how="left",
                suffixes=("", "_r2"),
            )
            merged = pd.concat([merged[~need], fb], axis=0).sort_index()

    if "senador_nombre" not in merged.columns:
        return hits_df

    def agg_detail(sub: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        sub2 = sub.dropna(subset=["senador_nombre"]).copy()
        sub2["senador_nombre"] = sub2["senador_nombre"].fillna("").astype(str)
        sub2 = sub2[sub2["senador_nombre"].str.strip() != ""]
        if sub2.empty:
            return ([], [])
        sub2["score_num"] = pd.to_numeric(sub2.get("score", 0), errors="coerce").fillna(0.0)
        sub2 = sub2.sort_values(["score_num"], ascending=False)

        names, details, seen = [], [], set()
        for _, rr in sub2.iterrows():
            nm = str(rr["senador_nombre"]).strip()
            if not nm or nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
            details.append({
                "senador": nm,
                "rol": str(rr.get("rol", "participa")),
                "score": float(pd.to_numeric(rr.get("score", 0), errors="coerce") or 0),
                "pl_norm": str(rr.get("pl_norm", "")),
            })
            if len(names) >= max_names:
                break
        return (names, details)

    out = merged.groupby("id", dropna=False).apply(agg_detail).reset_index(name="_tmp")
    out[out_col] = out["_tmp"].apply(lambda x: x[0])
    out[keep_detail_col] = out["_tmp"].apply(lambda x: x[1])
    out = out.drop(columns=["_tmp"])

    h2 = h.merge(out[["id", out_col, keep_detail_col]], on="id", how="left")

    # si ya venía m_senadores, combinar sin pisar
    if out_col in hits_df.columns:
        base = hits_df[[ "id", out_col ]].copy()
        base = base.rename(columns={out_col: f"{out_col}_base"})
        h2 = h2.merge(base, on="id", how="left")
        h2[out_col] = h2.apply(
            lambda rr: list(dict.fromkeys(_to_list_str(rr.get(f"{out_col}_base")) + _to_list_str(rr.get(out_col)))),
            axis=1,
        )
        h2 = h2.drop(columns=[f"{out_col}_base"])

    h2 = h2.drop(columns=["chunk_id_int"], errors="ignore")
    return h2

def senators_for_pl_from_roles(pl_norm: str, roles_df: Optional[pd.DataFrame], top_n: int = 8) -> List[str]:
    if roles_df is None or roles_df.empty or not pl_norm:
        return []
    if "pl_norm" not in roles_df.columns or "senador_nombre" not in roles_df.columns:
        return []
    sub = roles_df[roles_df["pl_norm"].astype(str) == str(pl_norm)].copy()
    if sub.empty:
        return []
    if "score" in sub.columns:
        sub = sub.sort_values("score", ascending=False)
    names = [s for s in sub["senador_nombre"].fillna("").astype(str).tolist() if s.strip()]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
        if len(out) >= top_n:
            break
    return out


# ======================================================================
# [BLOCK 04B] Directorio senadores (CSV) + fotos
# ======================================================================
@st.cache_data(show_spinner=False)
def safe_read_senadores_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        try:
            df = pd.read_csv(p, sep=";")
        except Exception as e:
            st.error(f"No pude leer senadores.csv: {path}\n{e}")
            return None

    # Normaliza columnas esperadas
    ren = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("nombre", "senador", "senador_nombre", "name"):
            ren[c] = "senador_nombre"
        elif cl in ("perfil_url", "profile_url", "url", "link"):
            ren[c] = "perfil_url"
        elif cl in ("foto", "foto_path", "photo", "photo_path", "imagen", "image"):
            ren[c] = "foto_path"
        elif cl in ("partido", "bancada", "party"):
            ren[c] = "partido"
        elif cl in ("departamento", "region", "circunscripcion"):
            ren[c] = "departamento"
    df = df.rename(columns=ren)

    if "senador_nombre" not in df.columns:
        return None

    for c in ["senador_nombre", "perfil_url", "foto_path", "partido", "departamento"]:
        if c not in df.columns:
            df[c] = ""

    df["senador_nombre"] = df["senador_nombre"].fillna("").astype(str)
    df["perfil_url"] = df["perfil_url"].fillna("").astype(str)
    df["foto_path"] = df["foto_path"].fillna("").astype(str)
    df["partido"] = df["partido"].fillna("").astype(str)
    df["departamento"] = df["departamento"].fillna("").astype(str)

    df = df[df["senador_nombre"].str.strip() != ""].copy()
    return df

def _norm_key_name(s: str) -> str:
    """
    Normaliza nombre a key:
    - lower
    - quita tildes
    - deja [a-z0-9]
    """
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def _strip_weird_img_suffix(filename: str) -> str:
    """
    Maneja casos tipo: Alex_Xavier_Fl_rez_Hern_ndezjpg  (sin punto antes de jpg)
    y también nombres normales con extensión.
    Retorna "base name" sin extensión real o pegada.
    """
    name = filename.strip()

    low = name.lower()

    # Caso normal: tiene extensión válida
    for ext in IMG_EXTS:
        if low.endswith(ext):
            return name[: -len(ext)]

    # Caso raro: termina en "jpg/jpeg/png/webp" sin el punto
    for ext in ["jpg", "jpeg", "png", "webp"]:
        if low.endswith(ext) and not low.endswith("." + ext):
            return name[: -len(ext)]

    # Si no detecta nada, devuelve tal cual
    return name

def build_photos_index(photos_dir: str) -> dict:
    """
    Indexa fotos por una key normalizada.
    - Soporta nombres con underscores.
    - Soporta extensiones .jpg/.jpeg/.png/.webp
    - Soporta casos donde 'jpg' está pegado al nombre sin punto.
    """
    pdir = Path(photos_dir)
    idx = {}

    if not pdir.exists():
        return idx

    for f in pdir.rglob("*"):
        if not f.is_file():
            continue

        fname = f.name
        low = fname.lower()

        # Acepta normales o raros sin punto
        ok = any(low.endswith(ext) for ext in IMG_EXTS) or any(low.endswith(x) for x in ["jpg", "jpeg", "png", "webp"])
        if not ok:
            continue

        base = _strip_weird_img_suffix(fname)
        base = base.replace("_", " ").strip()

        key = _norm_key_name(base)
        if key and key not in idx:
            idx[key] = str(f.resolve())

    return idx

def resolve_senator_profile_and_photo(
    senador_nombre: str,
    sen_dir_df: Optional[pd.DataFrame],
    photos_index: Dict[str, str],
    photos_dir: str,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Retorna:
      - perfil_url (si existe)
      - photo_path (si existe)
      - extra info dict (partido, departamento)
    Prioridad foto:
      1) foto_path en CSV (relativo a fotos_dir o absoluto)
      2) match por filename normalizado del nombre
    """
    perfil_url = ""
    extra = {"partido": "", "departamento": ""}
    photo_path = None
   

    sen_key = _norm_key_name(senador_nombre)
    photo_path = photos_index.get(sen_key)

    if not photo_path:
        nombre_tmp = senador_nombre
        for ch in ["é", "á", "í", "ó","ú"]:
            nombre_tmp = nombre_tmp.replace(ch, " ")
        sen_key2 = _norm_key_name(nombre_tmp)
        photo_path = photos_index.get(sen_key2)




    if sen_dir_df is not None and not sen_dir_df.empty:
        hit = sen_dir_df[sen_dir_df["senador_nombre"].astype(str) == str(senador_nombre)]
        if hit.empty:
            # fallback: matching por key normalizada
            key = _norm_key_name(senador_nombre)
            if key:
                tmp = sen_dir_df.copy()
                tmp["_k"] = tmp["senador_nombre"].astype(str).apply(_norm_key_name)
                hit = tmp[tmp["_k"] == key]
        if not hit.empty:
            r0 = hit.iloc[0]
            perfil_url = str(r0.get("perfil_url", "") or "")
            extra["partido"] = str(r0.get("partido", "") or "")
            extra["departamento"] = str(r0.get("departamento", "") or "")
            fp = str(r0.get("foto_path", "") or "").strip()
            if fp:
                p = Path(fp)
                if not p.is_absolute():
                    p = Path(photos_dir) / fp
                if p.exists():
                    photo_path = str(p)

    if photo_path is None:
        key = _norm_key_name(senador_nombre)
        photo_path = photos_index.get(key)

    return perfil_url, photo_path, extra
# ======================================================================
# [BLOCK 05] Themes (heurística si falta m_temas) + afinidad senador-tema
# ======================================================================
THEME_RULES = {
    "salud": ["salud", "hospital", "eps", "medic", "paciente", "clínic", "enfer","cancer","sida","Vejez"],
    "educación": ["educ", "univers", "coleg", "docente", "icfes", "aprendiz"],
    "seguridad": ["seguridad", "delito", "polic", "crimen", "violencia", "pena", "carcel"],
    "medio ambiente": ["ambient", "bosque", "agua", "residuo", "contamin", "clima", "fauna", "flora", "glifosato","flora","arbol"],
    "economía": ["econom", "impuesto", "tribut", "financ", "presupuesto", "invers", "empresa"],
    "trabajo": ["trabaj", "empleo", "salario", "laboral", "sindicat", "contrato"],
    "agro": ["agro", "campes", "cultivo", "tierra", "rural", "ganad", "papa", "café"],
    "justicia": ["justicia", "corte", "juez", "fiscal", "proceso", "penal", "civil"],
    "transporte": ["transporte", "vía", "carretera", "movilidad", "metro", "tránsito"],
}

def detect_themes_from_text(text: str, max_themes: int = 3) -> List[str]:
    t = (text or "").lower()
    hits = []
    for theme, kws in THEME_RULES.items():
        score = sum(1 for kw in kws if kw in t)
        if score > 0:
            hits.append((theme, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    if not hits:
        return ["SIN_TEMA"]
    return [h[0] for h in hits[:max_themes]]

def ensure_hits_have_themes(hits_df: pd.DataFrame) -> pd.DataFrame:
    if hits_df is None or hits_df.empty:
        return hits_df
    df = hits_df.copy()
    if "m_temas" not in df.columns:
        df["m_temas"] = None

    needs = df["m_temas"].isna() | (df["m_temas"].astype(str).str.strip() == "") | (df["m_temas"].astype(str).str.lower() == "nan")
    if needs.any():
        df.loc[needs, "m_temas"] = df.loc[needs, "text"].apply(lambda x: detect_themes_from_text(str(x)))

    # estandariza a lista
    df["m_temas"] = df["m_temas"].apply(lambda x: x if isinstance(x, list) else ([str(x)] if x is not None else ["SIN_TEMA"]))
    return df

ROLE_WEIGHTS = {"autor": 1.00, "ponente": 0.85, "coautor": 0.75, "firmante": 0.55}

def role_weight(rol: Any) -> float:
    """
    Rol -> peso. Robusto a NaN/float/int.
    """
    if rol is None:
        r = ""
    else:
        # pandas puede traer NaN como float
        try:
            if pd.isna(rol):
                r = ""
            else:
                r = str(rol)
        except Exception:
            r = str(rol)

    r = r.strip().lower()

    if "autor" in r:
        return 1.0
    if "ponente" in r:
        return 0.85
    if "coautor" in r or "co-autor" in r:
        return 0.75
    if "firm" in r:
        return 0.55
    return 0.40


def build_senator_theme_affinity(hits_df: pd.DataFrame, roles_df: Optional[pd.DataFrame], top_n: int = 2500) -> pd.DataFrame:
    """
    Afinidad senador<->tema basada en:
    - similarity = 1 - distance
    - lex_score
    - rol ponderado (si roles existe)
    """
    if hits_df is None or hits_df.empty:
        return pd.DataFrame(columns=["senador_nombre", "tema", "affinity_score", "pl_count", "hits_count"])

    df = ensure_hits_have_themes(hits_df)

    if "m_pl_norm" not in df.columns:
        df["m_pl_norm"] = "NA"

    tmp = df[["id", "m_pl_norm", "distance", "lex_score", "m_temas"]].copy()
    tmp["m_pl_norm"] = tmp["m_pl_norm"].fillna("NA").astype(str)
    tmp["similarity"] = 1.0 - pd.to_numeric(tmp["distance"], errors="coerce").fillna(1.0)
    tmp["lex_score"] = pd.to_numeric(tmp["lex_score"], errors="coerce").fillna(0.0)

    tmp = tmp.explode("m_temas").rename(columns={"m_temas": "tema"})
    tmp["tema"] = tmp["tema"].fillna("SIN_TEMA").astype(str)

    if roles_df is not None and not roles_df.empty:
        r = roles_df[["pl_norm", "senador_nombre", "rol", "score"]].copy() if "rol" in roles_df.columns else roles_df[["pl_norm", "senador_nombre", "score"]].copy()
        if "rol" not in r.columns:
            r["rol"] = "participa"
        r["pl_norm"] = r["pl_norm"].fillna("NA").astype(str)
        r["senador_nombre"] = r["senador_nombre"].fillna("NO_DISPONIBLE").astype(str)
        r["rol"] = r["rol"].fillna("participa").astype(str)
        r["score"] = pd.to_numeric(r["score"], errors="coerce").fillna(1.0)

        tmp = tmp.merge(r, how="left", left_on="m_pl_norm", right_on="pl_norm")
        tmp["senador_nombre"] = tmp["senador_nombre"].fillna("NO_DISPONIBLE").astype(str)
        tmp["role_w"] = tmp["rol"].apply(role_weight)
        tmp["score_w"] = tmp["score"]
    else:
        tmp["senador_nombre"] = "NO_DISPONIBLE"
        tmp["role_w"] = 0.40
        tmp["score_w"] = 1.0

    tmp["aff_part"] = (0.55 * tmp["similarity"] + 0.25 * tmp["lex_score"] + 0.20 * tmp["role_w"]) * tmp["score_w"]

    agg = tmp.groupby(["senador_nombre", "tema"], dropna=False).agg(
        affinity_score=("aff_part", "mean"),
        hits_count=("id", "count"),
        pl_count=("m_pl_norm", "nunique"),
    ).reset_index()

    agg = agg.sort_values(["affinity_score", "pl_count", "hits_count"], ascending=[False, False, False]).head(top_n)
    return agg

def normalize_0_100(vals: List[float]) -> List[float]:
    if not vals:
        return []
    v = np.array(vals, dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    mn, mx = float(v.min()), float(v.max())
    if mx - mn < 1e-9:
        return [50.0 for _ in vals]
    return [float(100.0 * (x - mn) / (mx - mn)) for x in v]

def plot_radar_theme_affinity(affinity_df: pd.DataFrame, senator: str, top_axes: int = 6) -> go.Figure:
    fig = go.Figure()
    if affinity_df is None or affinity_df.empty or not senator or senator in ("(elige)", "NO_DISPONIBLE"):
        fig.add_annotation(text="No hay datos para radar (faltan evidencias/roles).", x=0.5, y=0.5, showarrow=False)
        return ensure_plotly_fig_layout(fig, height=420)

    sub = affinity_df[affinity_df["senador_nombre"] == senator].copy()
    if sub.empty:
        fig.add_annotation(text="Este senador no tiene afinidad calculada aún.", x=0.5, y=0.5, showarrow=False)
        return ensure_plotly_fig_layout(fig, height=420)

    sub = sub.sort_values("affinity_score", ascending=False).head(top_axes)
    cats = sub["tema"].astype(str).tolist()
    raw = sub["affinity_score"].astype(float).tolist()
    vals = normalize_0_100(raw)

    cats2 = cats + [cats[0]]
    vals2 = vals + [vals[0]]

    fig.add_trace(go.Scatterpolar(
        r=vals2,
        theta=cats2,
        fill="toself",
        name=senator,
        hovertemplate="Tema: %{theta}<br>Proximidad (0-100): %{r:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Proximidad por temas (Radar) — basado en evidencias",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )
    return ensure_plotly_fig_layout(fig, height=420)

def plot_senator_center_graph(
    senator: str,
    affinity_df: pd.DataFrame,
    roles_df: Optional[pd.DataFrame],
    max_theme_nodes: int = 8,
    max_pl_nodes: int = 10,
) -> go.Figure:
    fig = go.Figure()
    if nx is None:
        fig.add_annotation(text="Grafo no disponible: instala networkx (pip install networkx).", x=0.5, y=0.5, showarrow=False)
        return ensure_plotly_fig_layout(fig, height=520)

    if not senator or senator in ("(elige)", "NO_DISPONIBLE"):
        fig.add_annotation(text="Selecciona un senador con datos.", x=0.5, y=0.5, showarrow=False)
        return ensure_plotly_fig_layout(fig, height=520)

    G = nx.Graph()
    G.add_node(senator, kind="senator")

    # temas top
    if affinity_df is not None and not affinity_df.empty:
        tsub = affinity_df[affinity_df["senador_nombre"] == senator].copy()
        tsub = tsub.sort_values("affinity_score", ascending=False).head(max_theme_nodes)
        for _, r in tsub.iterrows():
            theme = str(r["tema"])
            G.add_node(theme, kind="theme")
            G.add_edge(senator, theme, weight=float(r["affinity_score"]), rel="afinidad")

    # PLs top
    if roles_df is not None and not roles_df.empty and "senador_nombre" in roles_df.columns and "pl_norm" in roles_df.columns:
        rsub = roles_df[roles_df["senador_nombre"].astype(str) == str(senator)].copy()
        if not rsub.empty:
            rsub = rsub.sort_values("score", ascending=False) if "score" in rsub.columns else rsub
            pls = []
            for x in rsub["pl_norm"].fillna("").astype(str).tolist():
                if x.strip() and x not in pls:
                    pls.append(x)
                if len(pls) >= max_pl_nodes:
                    break
            for pl in pls:
                pl_node = f"PL: {pl}"
                G.add_node(pl_node, kind="pl")
                G.add_edge(senator, pl_node, weight=0.7, rel="participa")

    pos = nx.spring_layout(G, seed=7, k=0.75)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", opacity=0.35)

    node_x, node_y, node_text, node_size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        kind = G.nodes[n].get("kind", "other")
        deg = G.degree[n]
        size = 18 if kind == "senator" else (12 if kind == "theme" else 10)
        node_size.append(size + 1.5 * math.sqrt(max(deg, 1)))
        node_text.append(f"{n}<br>tipo={kind}<br>degree={deg}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[(n if G.nodes[n].get("kind") == "senator" else "") for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=node_size, opacity=0.9),
        hovertext=node_text,
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="Mapa centrado en senador (temas + PLs relacionados)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return ensure_plotly_fig_layout(fig, height=520)

# ======================================================================
# [BLOCK 06] Chroma Retriever
# ======================================================================
@st.cache_resource
def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

@st.cache_resource
def get_chroma_collection(persist_dir: str, collection: str):
    client = chromadb.PersistentClient(
        path=str(Path(persist_dir).resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(collection)

def chroma_query(
    col,
    embedder: SentenceTransformer,
    q: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
    fetch_k: int = 80,
    filter_bad_ocr: bool = True,
    lexical_rerank_on: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    where = where or {}
    fetch_k = max(fetch_k, k)
    t0 = time.perf_counter()
    t_emb0 = time.perf_counter()
    emb = embedder.encode([q])
    emb = normalize(np.array(emb), norm="l2")
    ms_embed = (time.perf_counter() - t_emb0) * 1000.0

    t_q0 = time.perf_counter()
    res = col.query(
        query_embeddings=emb.tolist(),
        n_results=fetch_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )
    if not isinstance(q, str):
        q = str(q)

    # Si por error embedder llega como string, lo detectamos y fallamos claro
    if not hasattr(embedder, "encode"):
        raise TypeError(f"embedder inválido: {type(embedder)} (¿orden de args incorrecto?)")

    ms_chroma = (time.perf_counter() - t_q0) * 1000.0

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    rows = []
    dropped_ocr = 0
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        doc = (doc or "").strip()
        doc_kind = None
        try:
            doc_kind = (meta or {}).get('doc_kind') or (meta or {}).get('m_doc_kind')
        except Exception:
            doc_kind = None
        if filter_bad_ocr and str(doc_kind).lower() not in ('plenaria','plenary') and looks_like_bad_ocr(doc):
            dropped_ocr += 1
            continue
        meta = meta or {}
        rows.append({
            "id": str(_id),
            "distance": float(dist),
            "similarity": 1.0 - float(dist),
            "text": doc,
            **{f"m_{mk}": mv for mk, mv in meta.items()}
        })

    df = pd.DataFrame(rows)
    if df.empty:
        dbg = {
            "q": q, "k": k, "where": where, "fetch_k": fetch_k,
            "dropped_ocr": dropped_ocr, "returned": 0,
            "ms_embed": round(ms_embed, 2),
            "ms_chroma": round(ms_chroma, 2),
            "ms_total": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        return df, dbg

    if lexical_rerank_on:
        df["lex_score"] = df["text"].apply(lambda t: lexical_score(q, t))
        df = df.sort_values(["lex_score", "distance"], ascending=[False, True]).reset_index(drop=True)
    else:
        df["lex_score"] = 0.0
        df = df.sort_values(["distance"], ascending=[True]).reset_index(drop=True)

    df = df.head(k).copy()

    dbg = {
        "q": q,
        "k": k,
        "where": where,
        "fetch_k": fetch_k,
        "filter_bad_ocr": filter_bad_ocr,
        "lexical_rerank": lexical_rerank_on,
        "post_filter_bad_ocr_dropped": dropped_ocr,
        "returned": int(len(df)),
        "ms_embed": round(ms_embed, 2),
        "ms_chroma": round(ms_chroma, 2),
        "ms_total": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return df, dbg

# ======================================================================
# [BLOCK 07] Sidebar (panel lateral)
# ======================================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    st.session_state["mode_expert"] = st.toggle("Modo experto", value=st.session_state["mode_expert"])
    st.session_state["ux_explain"] = st.toggle("Mostrar “Explícame fácil”", value=st.session_state["ux_explain"])

    st.divider()

    model_name = st.text_input("Embedding model", value=DEFAULT_MODEL)
    corpus_path = st.text_input("Corpus parquet", value=DEFAULT_CORPUS)
    chroma_dir = st.text_input("Chroma persist dir", value=DEFAULT_CHROMA_DIR)
    collection_name = st.text_input("Chroma collection", value=DEFAULT_COLLECTION)

    st.divider()
    st.subheader("🧱 Build (opcional)")

    colb1, colb2 = st.columns(2)
    if colb1.button("1) Build corpus RAG", use_container_width=True, ):
        cmd = [sys.executable, "tools/build_pl_rag_corpus.py"]
        rc, out = run_subprocess(cmd)
        st.session_state["_last_cmd"] = " ".join(cmd)
        st.session_state["_last_out"] = out
        st.session_state["_last_rc"] = rc

    if colb2.button("2) Build index Chroma", use_container_width=True, ):
        cmd = [sys.executable, "tools/build_pl_vector_index_chroma.py", "--recreate"]
        rc, out = run_subprocess(cmd)
        st.session_state["_last_cmd"] = " ".join(cmd)
        st.session_state["_last_out"] = out
        st.session_state["_last_rc"] = rc

    if st.button("Ver último log", use_container_width=True, ):
        st.session_state["_show_log"] = True

    if st.session_state.get("_show_log"):
        st.code(st.session_state.get("_last_cmd", ""))
        st.text_area("stdout+stderr", st.session_state.get("_last_out", ""), height=240)
        st.write("returncode:", st.session_state.get("_last_rc", None))

    st.divider()
    st.subheader("👤 Fuente de senadores")

    st.session_state["senator_source"] = st.radio(
        "¿De dónde quieres sacar los senadores?",
        ["Desde corpus (m_*)", "Desde roles_enriched.parquet (si existe)"],
        index=1 if st.session_state["senator_source"].startswith("Desde roles") else 0,
        help="Si tu corpus no trae m_senadores, usa roles_enriched para mostrarlos por chunk/pl."
    )

    st.divider()
    st.subheader("📦 Inputs opcionales")
    pl_text_canon_path = st.text_input("PL text canon parquet (proximity)", value=PL_TEXT_CANON)
    roles_enriched_path = st.text_input("Roles enriched parquet", value=ROLES_ENRICHED)

    # ---- defaults ANTES de instanciar widgets ----
    DEFAULT_SENADORES_CSV = r"data/HV_SENADORES/senadores.csv"  # ajusta a tu ruta real

    if "senadores_csv_path" not in st.session_state:
        st.session_state["senadores_csv_path"] = DEFAULT_SENADORES_CSV

    # ---- widget (NO reasignes session_state) ----
    senadores_csv_path = st.text_input(
        "Ruta senadores.csv",
        key="senadores_csv_path",
        help="CSV con links/perfil de senadores",
    )


   
    fotos_dir_path = st.text_input("Fotos senadores (carpeta)", value=DEFAULT_SENADORES_FOTOS_DIR, key="fotos_dir_path")

    st.divider()
    st.subheader("🎥 Plenarias + Links")

    plenarias_corpus_path = st.text_input(
        "Plenarias corpus parquet (RAG)",
        value=st.session_state.get("plenarias_corpus_path", DEFAULT_PLENARIAS_CORPUS),
        key="plenarias_corpus_path",
        help="Parquet con chunks de plenarias (meta: doc_kind, video_id, start_sec, plenaria_fecha, youtube_url, etc.).",
    )

    plenarias_chroma_dir = st.text_input(
        "Chroma persist_dir (Plenarias)",
        value=st.session_state.get("plenarias_chroma_dir", DEFAULT_PLENARIAS_CHROMA_DIR),
        key="plenarias_chroma_dir",
        help="Si indexaste plenarias en el mismo persist_dir que PL, deja igual.",
    )

    plenarias_collection = st.text_input(
        "Colección Chroma (Plenarias)",
        value=st.session_state.get("plenarias_collection", DEFAULT_PLENARIAS_COLLECTION),
        key="plenarias_collection",
    )

    search_scope = st.radio(
        "Buscar en",
        options=["Auto", "Proyectos de Ley", "Plenarias", "Ambos"],
        index=0,
        key="search_scope",
    )
    
# ======================================================================
# [BLOCK 08] Load data (corpus + roles)
# ======================================================================
df_corpus_raw = safe_read_parquet(corpus_path)
roles_df = safe_read_roles_enriched(roles_enriched_path)
df_plenarias_raw = safe_read_parquet(st.session_state.get("plenarias_corpus_path", DEFAULT_PLENARIAS_CORPUS))
senadores_df = safe_read_senadores_csv(st.session_state.get("senadores_csv_path", DEFAULT_SENADORES_CSV))
sen_dir_df = senadores_df
photos_dir = st.session_state.get("fotos_dir_path", DEFAULT_SENADORES_FOTOS_DIR)
photos_index = build_photos_index(photos_dir)

df_norm_for_themes = normalize_corpus_columns(df_corpus_raw) if df_corpus_raw is not None else None
themes_from_corpus = get_top_themes_from_corpus(df_norm_for_themes, top_n=60)
themes_options = ["(todos)"] + themes_from_corpus
if st.session_state["theme_selected"] not in themes_options:
    st.session_state["theme_selected"] = "(todos)"

# ======================================================================
# [BLOCK 09] Tabs (correctos)
# ======================================================================
tabs = st.tabs([
    "🔎 Buscar (RAG)",
    "📊 Tablero (Entendible)",
    "🧭 Explorar (Senadores/Temas)",
    "🕸️ Proximity (Avanzado)",
])

# ======================================================================
# [BLOCK 10] TAB 1 — Buscar (RAG) (control principal: worry + explain)
# ======================================================================
with tabs[0]:
    st.subheader("🔎 Buscar en Proyectos de Ley (RAG)")

    if df_corpus_raw is None:
        st.warning("No encuentro el corpus parquet. Revisa la ruta o construye el corpus.")
    else:
        st.caption(f"Corpus rows: {len(df_corpus_raw):,}")

    # Card principal (mobile-first)
    with st.container(border=True):
        st.markdown("#### ✨ Cuéntame en una frase")
        cA, cB = st.columns([3, 1])
        st.session_state["main_worry"] = cA.text_input(
            "¿Qué te preocupa hoy?",
            value=st.session_state["main_worry"],
            placeholder="Ej: glifosato, EPS, educación rural, seguridad...",
            label_visibility="collapsed",
            key="main_worry_input",
        )
        explain = cB.button("Explícame fácil", type="primary", use_container_width=True, )

        st.selectbox(
            "Tema (opcional)",
            options=themes_options,
            key="theme_selected",
            help="Filtra resultados por tema si m_temas está disponible; si no, no afecta.",
        )

        st.radio(
            "Evidencias a mostrar",
            ["Ambas (PL + Plenaria)", "Solo PL", "Solo Plenaria"],
            key="evidence_source",
            horizontal=True,
            help="Si ya indexaste plenarias en Chroma, aquí puedes separarlas visualmente."
        )

        # chips de filtros rápidos
        chip_cols = st.columns(3)
        st.session_state["filter_bad_ocr"] = chip_cols[0].toggle("🧹 Filtrar OCR malo", value=st.session_state["filter_bad_ocr"])
        st.session_state["lexical_rerank_on"] = chip_cols[1].toggle("🔤 Lexical rerank", value=st.session_state["lexical_rerank_on"])
        st.session_state["mode_expert"] = chip_cols[2].toggle("🧠 Modo experto", value=st.session_state["mode_expert"])

    # Filtros avanzados (expander)
    with st.expander("⚙️ Ajustes de búsqueda (avanzado)", expanded=False):
        c1, c2, c3 = st.columns([2.2, 1.0, 1.0])
        q = c1.text_input("Consulta", value=st.session_state["main_worry"])
        st.session_state["k"] = c2.slider("Top-k", 3, 25, st.session_state["k"])
        st.session_state["fetch_k"] = c3.slider("fetch_k", 20, 250, st.session_state["fetch_k"], step=10)

        c4, c5, c6 = st.columns(3)
        st.session_state["pl_norm_filter"] = c4.text_input("Filtro PL (pl_norm exacto)", value=st.session_state["pl_norm_filter"])
        st.session_state["section_filter"] = c5.selectbox(
            "Filtro sección",
            ["(todas)", "objeto", "exposicion_motivos", "articulado", "cuerpo", "firmantes", "tail"],
            index=["(todas)", "objeto", "exposicion_motivos", "articulado", "cuerpo", "firmantes", "tail"].index(st.session_state["section_filter"])
            if st.session_state["section_filter"] in ["(todas)", "objeto", "exposicion_motivos", "articulado", "cuerpo", "firmantes", "tail"] else 0
        )
        st.session_state["doc_kind_filter"] = c6.selectbox(
            "Filtro doc_kind",
            ["(todos)", "pl_base", "ponencia"],
            index=["(todos)", "pl_base", "ponencia"].index(st.session_state["doc_kind_filter"])
            if st.session_state["doc_kind_filter"] in ["(todos)", "pl_base", "ponencia"] else 0
        )
        st.session_state["show_text_chars"] = st.slider("Mostrar chars por chunk", 200, 3000, st.session_state["show_text_chars"], step=100)

    # Botón buscar (simple)
    bcols = st.columns([1, 3])
    do_search = bcols[0].button("Buscar", type="primary", use_container_width=True, )
    bcols[1].caption("Tip: si sale “sin resultados”, sube fetch_k o deja Tema=(todos).")

    where = {}
    if st.session_state["pl_norm_filter"].strip():
        where["pl_norm"] = st.session_state["pl_norm_filter"].strip()
    if st.session_state["section_filter"] != "(todas)":
        where["section"] = st.session_state["section_filter"]
    if st.session_state["doc_kind_filter"] != "(todos)":
        where["doc_kind"] = st.session_state["doc_kind_filter"]

   # Ejecutar búsqueda
    query_text = (st.session_state.get("main_worry") or "").strip()

    if do_search:
        # --- FIX: garantizar variables requeridas para el query (evita NameError) ---
        query_text = (query_text or "").strip()

        top_k = int(st.session_state.get("k") or 8)
        fetch_k = int(st.session_state.get("fetch_k") or 80)

        # usa el MISMO nombre que luego pasas al chroma_query
        lexical_rerank_on = bool(st.session_state.get("lexical_rerank_on", True))
        filter_bad_ocr = bool(st.session_state.get("filter_bad_ocr_on", st.session_state.get("filter_bad_ocr", True)))

        # scope efectivo (auto/pl/plenarias/ambos según tu UI)
        scope_eff = st.session_state.get("scope_mode", st.session_state.get("scope", "auto")) or "auto"

        try:
            embedder = get_embedder(model_name)

            # tu UI parece usar "search_scope" con valores: "Auto", "Proyectos de Ley", "Plenarias", "Ambos"
            scope_final = decide_scope(query_text, st.session_state.get("search_scope", "Auto"))

            # colección PL (principal)
            col_pl = get_chroma_collection(chroma_dir, collection_name)

            hits_list = []
            dbg_list = []

            # -----------------------------
            # PL
            # -----------------------------
            if scope_final in ("Proyectos de Ley", "Ambos"):
                hits_pl, dbg_pl = chroma_query(
                    col_pl,
                    embedder,
                    query_text,
                    top_k,
                    fetch_k=fetch_k,
                    where=where,
                    lexical_rerank_on=lexical_rerank_on,
                    # si tu chroma_query soporta esto, pásalo; si NO, bórralo:
                    filter_bad_ocr=filter_bad_ocr,
                )
                if hits_pl is not None and not hits_pl.empty:
                    hits_list.append(hits_pl)
                dbg_list.append(dbg_pl)

            # -----------------------------
            # PLENARIAS
            # -----------------------------
            if scope_final in ("Plenarias", "Ambos"):
                col_plen = get_chroma_collection(
                    st.session_state.get("plenarias_chroma_dir", DEFAULT_PLENARIAS_CHROMA_DIR),
                    st.session_state.get("plenarias_collection", DEFAULT_PLENARIAS_COLLECTION),
                )

                # En plenarias: no aplican filtros por tema/PL => where={}
                hits_plen, dbg_plen = chroma_query(
                    col_plen,
                    embedder,
                    query_text,
                    top_k,
                    fetch_k=fetch_k,
                    where={},
                    lexical_rerank_on=lexical_rerank_on,
                    filter_bad_ocr=filter_bad_ocr,
                )
                if hits_plen is not None and not hits_plen.empty:
                    hits_list.append(hits_plen)
                dbg_list.append(dbg_plen)

            hits_df = pd.concat(hits_list, ignore_index=True) if hits_list else pd.DataFrame()
            dbg = dbg_list[-1] if dbg_list else None

            # enriquecer senadores desde roles
            if st.session_state.get("senator_source", "").startswith("Desde roles"):
                hits_df = attach_senators_from_roles(
                    hits_df=hits_df,
                    roles_df=roles_df,
                    prefer_chunk_id=True,
                    out_col="m_senadores",
                    keep_detail_col="m_senadores_roles_detail",
                    max_names=10,
                )

            # asegurar temas si faltan
            hits_df = ensure_hits_have_themes(hits_df)

            # post-filter por tema
            hits_df = filter_hits_by_theme(hits_df, st.session_state.get("theme_selected"))

            # separar PL vs Plenaria por doc_kind según "evidence_source"
            if isinstance(hits_df, pd.DataFrame) and (not hits_df.empty) and ("m_doc_kind" in hits_df.columns):
                src = st.session_state.get("evidence_source", "Ambas (PL + Plenaria)")
                dk = hits_df["m_doc_kind"].fillna("NA").astype(str).str.lower()
                is_plen = dk.str.contains("plenaria")

                if src == "Solo PL":
                    hits_df = hits_df[~is_plen].reset_index(drop=True)
                elif src == "Solo Plenaria":
                    hits_df = hits_df[is_plen].reset_index(drop=True)

            st.session_state["hits_df"] = hits_df
            st.session_state["hits_dbg"] = dbg

        except Exception as e:
            st.error(f"Error en query: {e}")

    hits_df = st.session_state.get("hits_df")
    dbg = st.session_state.get("hits_dbg")


    # panel performance
    if isinstance(dbg, dict):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("ms embed", dbg.get("ms_embed", "—"))
        p2.metric("ms chroma", dbg.get("ms_chroma", "—"))
        p3.metric("ms total", dbg.get("ms_total", "—"))
        p4.metric("hits", dbg.get("returned", 0))

        if psutil is not None:
            try:
                mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
                st.caption(f"RAM proceso (aprox): {mem:.1f} MB")
            except Exception:
                pass

        if st.session_state["mode_expert"]:
            st.caption(f"debug: {dbg}")

    # Respuesta “Explícame fácil”
    if explain and isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
        with st.container(border=True):
            st.markdown("#### 🧾 Explicación fácil (basada en evidencia recuperada)")
            top = hits_df.iloc[0]
            pl = str(top.get("m_pl_norm", "NA"))
            temas = _to_list_str(top.get("m_temas"))
            sen = _to_list_str(top.get("m_senadores"))
            sen_txt = render_senadores_links(sen, sen_dir_df) if sen else "No disponible"
            tema_txt = ", ".join(temas) if temas else "SIN_TEMA"
            st.write(f"**PL más cercano:** {pl}")
            st.write(f"**Temas:** {tema_txt}")
            st.markdown(f"**Senadores (según roles/evidencia):** {sen_txt}")
            st.write("**Qué dice la evidencia (resumen corto):**")
            t = (top.get("text", "") or "").strip()
            st.write((t[:600] + "…") if len(t) > 600 else t)

    # Resultados
    if isinstance(hits_df, pd.DataFrame) and not hits_df.empty:
        # ============================================================
        # TAB 1 - RESUMEN "FÁCIL" (qué se encontró)
        # ============================================================
        st.success(f"Encontré {len(hits_df)} evidencias (chunks) relacionadas con tu consulta.")

        # Asegura columnas base para no romper
        for col in ["m_pl_norm", "m_section", "m_doc_kind", "m_doc_file", "m_senadores", "m_temas"]:
            if col not in hits_df.columns:
                hits_df[col] = None


        # Separación visual: PL vs Plenaria (según doc_kind)
        dk = hits_df["m_doc_kind"].fillna("NA").astype(str).str.lower()
        is_plen = dk.str.contains("plenaria")
        hits_pl = hits_df[~is_plen].reset_index(drop=True)
        hits_plen = hits_df[is_plen].reset_index(drop=True)

        if not hits_plen.empty:
            st.info(f"🎥 También encontré {len(hits_plen)} evidencias de **Plenaria** (puedes filtrarlas arriba).")

        # Agrupación por PL
        grp = hits_pl.groupby("m_pl_norm", dropna=False).agg(
            hits=("id", "count"),
            min_distance=("distance", "min"),
            best_lex=("lex_score", "max"),
            sections=("m_section", lambda x: list(dict.fromkeys([str(i) for i in x.fillna("NA").tolist() if str(i).strip()]))[:6]),
        ).reset_index()

        grp["m_pl_norm"] = grp["m_pl_norm"].fillna("NA").astype(str)
        grp = grp.sort_values(["hits", "min_distance"], ascending=[False, True]).reset_index(drop=True)

        # ---------- PANEL SUPERIOR: KPIs + explicación
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("PL encontrados", int(grp["m_pl_norm"].nunique()))
        kpi2.metric("Evidencias", int(len(hits_df)))
        kpi3.metric("Mejor cercanía (min dist)", f"{float(grp['min_distance'].min()):.4f}")
        kpi4.metric("Confianza", confidence_label(float(grp["min_distance"].min())))

        st.caption("💡 *Interpretación simple:* más evidencias por PL + menor distancia ⇒ más probable que ese PL sea relevante.")

        # ---------- GRÁFICOS SIMPLES (más útiles que hist/box de distancia)
        g1, g2, g3 = st.columns(3)

        # Top PLs por evidencia
        topN = min(12, len(grp))
        pl_df = grp.head(topN).copy().sort_values("hits", ascending=True)
        fig_pl = px.bar(pl_df, x="hits", y="m_pl_norm", orientation="h", title="Top PLs por evidencias")
        show_fig(fig_pl, 360, "tab1_top_pls")

        # Secciones en hits
        sec_counts = hits_df["m_section"].fillna("NA").astype(str).value_counts().head(10)
        sec_df = sec_counts.reset_index()
        sec_df.columns = ["section", "count"]
        fig_sec = px.bar(sec_df.sort_values("count", ascending=True), x="count", y="section", orientation="h", title="Dónde aparece (sección)")
        with g2:
            show_fig(fig_sec, 360, "tab1_sections")

        # Senadores (si existen)
        if hits_df["m_senadores"].notna().any():
            tmp_sen = explode_listlike(hits_df[["m_senadores"]].copy(), "m_senadores")
            sen_counts = tmp_sen["m_senadores"].fillna("NA").astype(str).value_counts().head(10)
            sen_df = sen_counts.reset_index()
            sen_df.columns = ["senador", "count"]
            fig_sen = px.bar(sen_df.sort_values("count", ascending=True), x="count", y="senador", orientation="h", title="Senadores más mencionados")
            with g3:
                show_fig(fig_sen, 360, "tab1_senadores")
        else:
            with g3:
                st.info("Aún no tengo senadores en estos hits. Activa fuente roles_enriched o verifica roles_df.")

        st.divider()

        # ---------- CONTROL PRINCIPAL (tu token obligatorio)
        cQ, cB = st.columns([3, 1])
        user_worry = cQ.text_input("¿Qué te preocupa hoy?", value=q, key="main_worry")
        explain_easy = cB.button("Explícame fácil", type="primary", use_container_width=True, key="btn_explain_easy_tab1")


        if explain_easy:
            # mini resumen narrativo
            top_pl = grp.iloc[0]["m_pl_norm"]
            top_hits = int(grp.iloc[0]["hits"])
            top_dist = float(grp.iloc[0]["min_distance"])

            st.markdown("## 🧠 Explicación fácil")
            st.write(
                f"1) Lo que más aparece en tu búsqueda es el **PL: {top_pl}**.\n\n"
                f"2) Tengo **{top_hits} evidencias** para ese PL.\n\n"
                f"3) La mejor cercanía es **{top_dist:.4f}** (confianza: **{confidence_label(top_dist)}**)."
            )
            st.caption("Si quieres, dime: *“¿qué significa este PL?”* o *“quiénes lo impulsan?”* y lo resumo por secciones.")

        st.divider()

        # ---------- CARDS POR PL (mobile-first)
        st.subheader("🧾 Resultados por Proyecto de Ley")
        show_pl_cards = st.slider("Cuántos PL mostrar", 3, 25, min(10, len(grp)), key="tab1_cards_n")

        for _, row in grp.head(show_pl_cards).iterrows():
            pl = str(row["m_pl_norm"])
            sub = hits_pl[hits_pl["m_pl_norm"].astype(str) == pl].copy()

            best = pick_best_evidence(sub)
            min_dist = float(row["min_distance"])
            conf = confidence_label(min_dist)

            # senadores: usa hit si existe; si no, fallback por PL desde roles
            sen_txt = "—"
            if best is not None:
                sen_val = best.get("m_senadores", None)
                sen_list = _to_list_str(sen_val)
                if not sen_list:
                    sen_list = senators_for_pl_from_roles(pl, roles_df, top_n=8)
                sen_txt = render_senadores_links(sen_list, sen_dir_df) if sen_list else "No disponible"

            # temas
            tema_txt = "—"
            if best is not None:
                tema_txt = list_to_text(best.get("m_temas", None), max_items=6)

            # resumen
            snippet = ""
            if best is not None:
                snippet = (best.get("text") or "").strip()
            if len(snippet) > 520:
                snippet = snippet[:520] + "…"

            with st.container(border=True):
                h1, h2, h3 = st.columns([2.1, 0.9, 1.0])
                h1.markdown(f"### {pl}")
                # chip: tipo evidencia predominante
                dk = sub.get("m_doc_kind", pd.Series(["NA"])).fillna("NA").astype(str).value_counts().index[0]
                h1.caption(f"Tipo evidencia: {dk}")
                h2.metric("Evidencias", int(row["hits"]))
                h3.metric("Confianza", conf)

                i1, i2, i3 = st.columns(3)
                i1.write("**👤 Senadores**")
                # render links si tenemos directorio
                if senadores_df is not None and sen_list:
                    for nm in sen_list[:8]:
                        url = get_senator_profile_url(nm, senadores_df)
                        if url:
                            i1.markdown(f"- [{nm}]({url})")
                        else:
                            i1.write(f"- {nm}")
                else:
                    i1.write(sen_txt)
                i2.write("**🧩 Temas**"); i2.write(tema_txt)
                i3.write("**📌 Secciones**"); i3.write(", ".join(row["sections"]) if row.get("sections") else "—")

                if snippet:
                    st.write("**Resumen (mejor evidencia):**")
                    st.write(snippet)

                # Evidencia detallada (no satura)
                with st.expander("Ver evidencias (top del PL)", expanded=False):
                    ev_cols = ["id", "distance", "lex_score", "m_section", "m_doc_kind", "m_doc_file", "text"]
                    ev_cols = [c for c in ev_cols if c in sub.columns]

                    # etiquetas rápidas: doc_kind + youtube
                    sub_show = sub[ev_cols].copy()
                    if "m_doc_kind" in sub_show.columns:
                        sub_show["doc_kind"] = sub_show["m_doc_kind"].fillna("NA").astype(str)
                    else:
                        sub_show["doc_kind"] = "NA"

                    sub_show["youtube_url"] = sub_show.apply(youtube_url_from_meta, axis=1)

                    ev_cols2 = ["id", "doc_kind", "m_section", "distance", "lex_score", "youtube_url", "text"]
                    ev_cols2 = [c for c in ev_cols2 if c in sub_show.columns]

                    st.dataframe(
                        sub_show[ev_cols2].sort_values(["distance", "lex_score"], ascending=[True, False]).head(8),
                        use_container_width=True,
                        height=280,
                    )

                    

                    best_url = youtube_url_from_meta(
                        sub_show.sort_values(["distance", "lex_score"], ascending=[True, False]).iloc[0]
                    )



        # ------------------------------------------------------------
        # 🎥 Evidencias de Plenaria (cards por fecha/video)
        # ------------------------------------------------------------
        if not hits_plen.empty:
            st.subheader("🎥 Evidencias de Plenaria")
            # asegura columnas
            for c in ["m_plenaria_fecha", "m_video_id", "m_start_sec", "m_doc_file", "m_section"]:
                if c not in hits_plen.columns:
                    hits_plen[c] = None

            # agrupación por fecha + video
            hits_plen["plenaria_key"] = (
                hits_plen["m_plenaria_fecha"].fillna("SIN_FECHA").astype(str)
                + " | "
                + hits_plen["m_video_id"].fillna("SIN_VIDEO").astype(str)
            )

            gpl = hits_plen.groupby("plenaria_key", dropna=False).agg(
                hits=("id", "count"),
                min_distance=("distance", "min"),
            ).reset_index().sort_values(["hits", "min_distance"], ascending=[False, True])

            show_n = st.slider("Cuántas plenarias mostrar", 1, 20, min(6, len(gpl)), key="tab1_plen_cards_n")
            for _, rr in gpl.head(show_n).iterrows():
                key = str(rr["plenaria_key"])
                subp = hits_plen[hits_plen["plenaria_key"] == key].copy().sort_values(["distance", "lex_score"], ascending=[True, False])
                bestp = subp.iloc[0]
                fecha = str(bestp.get("m_plenaria_fecha", "SIN_FECHA"))
                vid = bestp.get("m_video_id", None)
                ss = bestp.get("m_start_sec", 0)
                url = youtube_url(vid, ss)

                with st.container(border=True):
                    h1, h2, h3 = st.columns([2.2, 0.9, 1.0])
                    h1.markdown(f"### {fecha}")
                    h2.metric("Evidencias", int(rr["hits"]))
                    h3.metric("Confianza", confidence_label(float(rr["min_distance"])))

                    if url:
                        st.link_button("▶️ Verificar en YouTube (desde este minuto)", url, use_container_width=True, )

                    snippet = (bestp.get("text") or "").strip()
                    if len(snippet) > 520:
                        snippet = snippet[:520] + "…"
                    if snippet:
                        st.write("**Qué se dijo (evidencia top):**")
                        st.write(snippet)

                    with st.expander("Ver evidencias (top de la plenaria)", expanded=False):
                        ev_cols = ["id", "distance", "lex_score", "m_start_sec", "m_section", "m_doc_file", "text"]
                        ev_cols = [c for c in ev_cols if c in subp.columns]
                        st.dataframe(subp[ev_cols].head(10), use_container_width=True, height=280)

    else:
        if dbg is not None:
            st.info("Sin resultados (o filtrados por OCR/tema). Prueba subir fetch_k o poner Tema=(todos).")


# ======================================================================
# [BLOCK 11] TAB 2 — Tablero (Entendible)  ✅ (tabs[1])
# ======================================================================
with tabs[1]:
    st.subheader("📊 Tablero (Entendible)")
    st.caption("Qué PL sale, de qué trata, quién está detrás (según evidencia) y qué tan fuerte fue el match.")

    if df_corpus_raw is None:
        st.warning("Sin corpus no hay tablero global.")
        st.stop()

    df = normalize_corpus_columns(df_corpus_raw)
    hits_df = st.session_state.get("hits_df")
    has_hits = isinstance(hits_df, pd.DataFrame) and not hits_df.empty

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chunks en corpus", f"{len(df):,}")
    k2.metric("PL únicos", f"{df['m_pl_norm'].nunique():,}" if "m_pl_norm" in df.columns else "—")
    k3.metric("Secciones", f"{df['m_section'].nunique():,}" if "m_section" in df.columns else "—")
    k4.metric("Hits actuales", f"{len(hits_df):,}" if has_hits else "0")

    st.divider()

    left, right = st.columns(2)
    if "m_section" in df.columns:
        gsec = df["m_section"].fillna("NA").astype(str).value_counts().head(12)
        gsec_df = gsec.reset_index()
        gsec_df.columns = ["section", "count"]
        with left:
            show_fig(px.bar(gsec_df, x="section", y="count", title="Global: chunks por sección"), 360, "global_section")

    if "m_doc_kind" in df.columns:
        gdk = df["m_doc_kind"].fillna("NA").astype(str).value_counts()
        gdk_df = gdk.reset_index()
        gdk_df.columns = ["doc_kind", "count"]
        with right:
            show_fig(px.pie(gdk_df, names="doc_kind", values="count", title="Global: doc_kind"), 360, "global_dockind")

    st.divider()

    st.subheader("🧾 Resultados actuales por PL (cards)")
    if not has_hits:
        st.info("Ejecuta una búsqueda en la pestaña 🔎 Buscar (RAG).")
    else:
        # asegurar columnas esperadas
        for base in ["m_pl_norm", "m_section", "m_doc_kind", "m_doc_file", "m_senadores", "m_partidos_unicos", "m_comisiones", "m_temas"]:
            if base not in hits_df.columns:
                hits_df[base] = None

        grp = hits_pl.groupby("m_pl_norm", dropna=False).agg(
            hits=("id", "count"),
            min_distance=("distance", "min"),
        ).reset_index().sort_values(["hits", "min_distance"], ascending=[False, True])

        show_pl_cards = st.slider("Cuántos PL mostrar", 3, 25, 10, key="pl_cards_n")

        preferred_sections = ["objeto", "exposicion_motivos", "cuerpo", "articulado"]

        for _, row in grp.head(show_pl_cards).iterrows():
            pl = str(row["m_pl_norm"])
            sub = hits_df[hits_df["m_pl_norm"] == pl].copy()
            sub = sub.sort_values(["distance", "lex_score"], ascending=[True, False])

            best = None
            for s in preferred_sections:
                cand = sub[sub["m_section"].astype(str) == s]
                if len(cand) > 0:
                    best = cand.iloc[0]
                    break
            if best is None:
                best = sub.iloc[0]

            # ✅ senadores bonito (lista/string) + fallback PL->roles
            sen_val = best.get("m_senadores", None)
            sen_list = _to_list_str(sen_val)
            if not sen_list:
                sen_list = senators_for_pl_from_roles(pl, roles_df, top_n=8)
            sen_txt = render_senadores_links(sen_list, sen_dir_df) if sen_list else "No disponible"

            par = best.get("m_partidos_unicos", None)
            com = best.get("m_comisiones", None)
            tem_list = _to_list_str(best.get("m_temas", None))
            tem_txt = ", ".join(tem_list) if tem_list else "SIN_TEMA"

            with st.container(border=True):
                hcol1, hcol2, hcol3 = st.columns([2.2, 1.0, 1.0])
                hcol1.markdown(f"### {pl}")
                hcol2.metric("hits", int(row["hits"]))
                hcol3.metric("min dist", f"{float(row['min_distance']):.4f}")

                info_cols = st.columns(4)
                info_cols[0].write("**👤 Senadores**")
                if senadores_df is not None and sen_list:
                    for nm in sen_list[:8]:
                        url = get_senator_profile_url(nm, senadores_df)
                        if url:
                            info_cols[0].markdown(f"- [{nm}]({url})")
                        else:
                            info_cols[0].write(f"- {nm}")
                else:
                    info_cols[0].write(sen_txt)
                info_cols[1].write("**🏛️ Partidos**"); info_cols[1].write(str(par) if is_present_value(par) else "—")
                info_cols[2].write("**📌 Comisión**"); info_cols[2].write(str(com) if is_present_value(com) else "—")
                info_cols[3].write("**🧩 Temas**"); info_cols[3].write(tem_txt)

                resumen = (best.get("text") or "").strip()
                if len(resumen) > 900:
                    resumen = resumen[:900] + "…"
                st.write("**Resumen (evidencia top):**")
                st.write(resumen)

                with st.expander("Ver evidencias (top chunks de este PL)", expanded=False):
                    ev_cols = ["id", "distance", "m_section", "m_doc_kind", "m_doc_file", "lex_score", "text"]
                    ev_cols = [c for c in ev_cols if c in sub.columns]

                    sub_show = sub[ev_cols].copy()
                    if "m_doc_kind" in sub_show.columns:
                        sub_show["doc_kind"] = sub_show["m_doc_kind"].fillna("NA").astype(str)
                    else:
                        sub_show["doc_kind"] = "NA"

                    sub_show["youtube_url"] = sub_show.apply(youtube_url_from_meta, axis=1)

                    ev_cols2 = ["id", "doc_kind", "m_section", "distance", "lex_score", "youtube_url", "text"]
                    ev_cols2 = [c for c in ev_cols2 if c in sub_show.columns]

                    st.dataframe(
                        sub_show[ev_cols2].head(8), use_container_width=True,
                        height=260,
                    )

                    best_url = youtube_url_from_meta(sub_show.iloc[0]) if len(sub_show) else None
                    if best_url:
                        st.write("🔗 Verificación:", best_url)


# ======================================================================
# [BLOCK 12] TAB 3 — Explorar (Senadores/Temas) ✅ (tabs[2])
# ======================================================================
with tabs[2]:
    st.subheader("🧭 Explorar (Senadores / Temas)")
    st.caption("Aquí sí ves senadores: desde roles_enriched y afinidad por temas basada en evidencias.")

    hits_df = st.session_state.get("hits_df")
    if not (isinstance(hits_df, pd.DataFrame) and not hits_df.empty):
        st.info("Primero ejecuta una búsqueda en 🔎 Buscar (RAG).")
    elif roles_df is None or roles_df.empty:
        st.warning("No se pudo cargar roles_enriched.parquet. Revisa la ruta en el panel lateral.")
    else:
        # asegurar temas
        hits_df = ensure_hits_have_themes(hits_df)

        # afinidad
        affinity_df = build_senator_theme_affinity(hits_df, roles_df, top_n=2500)

        # selector senador
        sen_list = sorted([s for s in roles_df["senador_nombre"].dropna().astype(str).unique().tolist() if s.strip()])
        if not sen_list:
            st.warning("roles_enriched no trae senador_nombre usable.")
        else:
            s1, s2 = st.columns([2, 1])
            senator = s1.selectbox("Selecciona un senador", ["(elige)"] + sen_list, index=0)
            top_axes = s2.slider("Top temas (radar)", 3, 10, 6)

            if senator != "(elige)":

                # --- Perfil senador (foto + link) ---
                perfil_url, photo_path, extra = resolve_senator_profile_and_photo(
                    senador_nombre=senator,
                    sen_dir_df=senadores_df if 'senadores_df' in globals() else None,
                    photos_index=photos_index if 'photos_index' in globals() else {},
                    photos_dir=photos_dir if 'photos_dir' in globals() else st.session_state.get('fotos_dir_path', DEFAULT_SENADORES_FOTOS_DIR),
                )

                with st.container(border=True):
                    a1, a2 = st.columns([1, 3])
                    with a1:
                        if photo_path and Path(photo_path).exists():
                            st.image(photo_path, width=160)
                        else:
                            st.caption("📷 Sin foto local para este senador.")
                    with a2:
                        st.markdown(f"#### {senator}")
                        if extra.get("partido") or extra.get("departamento"):
                            st.caption(f"🏛️ {extra.get('partido','')}".strip() + (f" · 📍 {extra.get('departamento','')}" if extra.get('departamento') else ""))
                        if perfil_url:
                            st.markdown(f"🔗 Perfil: [{perfil_url}]({perfil_url})")
                        else:
                            st.caption("🔗 Sin perfil_url en senadores.csv")

                c1, c2 = st.columns(2)
                with c1:
                    show_fig(plot_radar_theme_affinity(affinity_df, senator, top_axes=top_axes), 420, "radar")
                with c2:
                    show_fig(plot_senator_center_graph(senator, affinity_df, roles_df), 520, "sen_graph")

                st.divider()

                st.markdown("#### 📌 Top temas del senador (según evidencia)")
                sub = affinity_df[affinity_df["senador_nombre"] == senator].copy()
                sub = sub.sort_values(["affinity_score", "pl_count", "hits_count"], ascending=[False, False, False]).head(25)
                st.dataframe(
                    sub,
                    use_container_width=True,
                    height=280,
                )

                
                

                st.markdown("#### 🗂️ PLs del senador (desde roles_enriched)")
                rsub = roles_df[roles_df["senador_nombre"].astype(str) == str(senator)].copy()
                rsub = rsub.sort_values("score", ascending=False) if "score" in rsub.columns else rsub
                cols_show = [c for c in ["pl_norm", "rol", "score", "section", "doc_kind", "doc_file"] if c in rsub.columns]
                st.dataframe(
                    rsub[cols_show].head(60),
                    use_container_width=True,
                    height=280,
                )


# ======================================================================
# [BLOCK 13] TAB 4 — Proximity (Avanzado) ✅ (tabs[3]) (placeholder simple)
# ======================================================================
with tabs[3]:
    st.subheader("🕸️ Proximity (Avanzado)")
    st.caption("Si luego quieres red PL↔PL, aquí conectas PL_TEXT_CANON y un kNN. (Lo dejamos listo para extender).")

    df_pl_text = safe_read_parquet(pl_text_canon_path)
    if df_pl_text is None or df_pl_text.empty:
        st.warning("No encuentro pl_text_canon parquet. Revisa la ruta en el panel lateral.")
    else:
        st.info("✅ Dataset de texto por PL encontrado. Si quieres, te armo el kNN + grafo completo en este tab.")

# ======================================================================
# [BLOCK 14] Footer
# ======================================================================
with st.expander("📌 Notas rápidas", expanded=False):
    st.markdown(
        """
- Si ves textos “basura”, activa **Filtrar OCR malo** y sube **fetch_k** (80–200) + **Lexical rerank**.
- **Senadores**: si tu corpus no trae `m_senadores`, este app los saca de `roles_enriched.parquet` (y muestra evidencia rol/score).
- El error **“truth value of an empty array…”** queda corregido con `is_present_value()` (no usamos `pd.notna()` sobre listas/arrays).
        """
    

)