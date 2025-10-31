#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P08 — Análisis de Productos de Combustión

Este script guía el análisis de productos de combustión, siguiendo las
indicaciones de la práctica (PDF) y complementando con análisis estadístico,
detección de picos, y cálculos estequiométricos (AFR, λ, φ).

Se guardarán todos los resultados (CSV) e imágenes (PNG y SVG) en
la carpeta `data/`.

Estructura de carpetas esperada:
Tu_Proyecto/
├── src/
│   └── p08_script.py  (Este script)
└── data/
    ├── tu_archivo_de_datos_1.csv (Entrada)
    ├── processed_combustion_data.csv (Salida)
    └── figures/ (Salida)

Notas:
- Coloca tus archivos medidos (CSV/TXT/TSV) en `data/`.
- Si no se encuentran archivos, se generará un conjunto de datos sintético.
"""

# %% [setup] Importaciones y utilidades
import os
import re
import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Estilo de gráficos
sns.set(context="notebook", style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})

# Directorio de datos
# Asumimos que el script está en `.../src/p08_script.py`
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# MODIFICADO: La salida estará en data/ al mismo nivel que src/
DATA_DIR = (SCRIPT_DIR.parent / "data").resolve()


# %% Helper para guardar figuras en PNG y SVG
def save_fig(fig: plt.Figure, name: str, subdir: str = "figures", tight: bool = True):
    """Guarda una figura de matplotlib en PNG y SVG."""
    out_dir = DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name.replace(" ", "_").replace("/", "-")
    png = out_dir / f"{safe}.png"
    svg = out_dir / f"{safe}.svg"
    if tight:
        fig.tight_layout()
    fig.savefig(png)
    fig.savefig(svg)
    print(f"Guardado: {png} y {svg}")

# Utilidad básica de reporte
def print_status(msg: str):
    """Imprime un mensaje con timestamp."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


# %% [data] Funciones de carga flexible de archivos y consolidación

# Columnas canónicas
CANON = {
    "time": ["time", "tiempo", "t", "timestamp", "date", "datetime"],
    "O2": ["o2", "o2_%", "o2[%]", "o2_vol", "o2_vol%"],
    "CO2": ["co2", "co2_%", "co2[%]", "co2_vol", "co2_vol%"],
    "CO": ["co", "co_ppm", "co[ppm]", "co_mg/m3", "co_mg_m3"],
    "NOx": ["nox", "no+no2", "nox_ppm", "nox[ppm]"],
    "T": ["t", "temp", "temperature", "tgas", "tg"],
    "P": ["p", "presion", "pressure"],
}

# Mapeo inverso para normalizar nombres
def build_name_map():
    name_map = {}
    for canon, variants in CANON.items():
        for v in variants:
            name_map[v.lower()] = canon
        name_map[canon.lower()] = canon
    return name_map

NAME_MAP = build_name_map()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        key = re.sub(r"\s+", "", str(c)).lower()
        new_cols.append(NAME_MAP.get(key, str(c)))
    df = df.copy()
    df.columns = new_cols
    return df

def read_any(path: Path) -> Optional[pd.DataFrame]:
    try:
        # Auto separador con engine=python y sep=None
        df = pd.read_csv(path, engine="python", sep=None)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print_status(f"No se pudo leer {path.name}: {e}")
            return None
    df = normalize_columns(df)
    # Manejo de tiempo
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        except Exception:
            pass
    else:
        df["time"] = pd.RangeIndex(len(df))
    df["source_file"] = path.name
    return df

def scan_data_files(folder: Path, exts: Tuple[str, ...] = (".csv", ".txt", ".tsv")) -> List[Path]:
    """
    Escanea el directorio en busca de archivos de datos, excluyendo
    los archivos de salida conocidos generados por este propio script.
    """
    # Archivos de salida a excluir para no leerlos como entrada
    outputs_to_exclude = {
        "processed_combustion_data.csv",
        "synthetic_example_combustion.csv",
        "combustion_metrics.csv",
        "temperatura_adiabatica.csv",
        "stats_descriptivos.csv",
        "stats_outliers_iqr.csv",
        "stats_rolling_30.csv",
        "peaks_CO.csv",
        "peaks_NOx.csv",
        "peaks_T.csv",
        "summary_maximos.json",
        "outputs_manifest.json"
    }
    
    # Excluir también los archivos de diagnóstico
    outputs_to_exclude.add("peaks_CO_diagnostic.csv")
    outputs_to_exclude.add("peaks_NOx_diagnostic.csv")
    outputs_to_exclude.add("peaks_T_diagnostic.csv")


    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))

    # Filtrar archivos de salida
    filtered_files = [f for f in files if f.name not in outputs_to_exclude]
    return sorted(filtered_files)

def load_data(data_dir: Path) -> pd.DataFrame:
    """Carga y consolida archivos de datos o genera datos sintéticos."""
    files = scan_data_files(data_dir)
    print_status(f"Archivos de entrada detectados: {[f.name for f in files]}")

    frames = []
    for f in files:
        df = read_any(f)
        if df is not None and len(df) > 0:
            frames.append(df)

    if frames:
        raw = pd.concat(frames, ignore_index=True, sort=False)
        # Orden temporal si es posible
        if pd.api.types.is_datetime64_any_dtype(raw.get("time")):
            raw = raw.sort_values("time").reset_index(drop=True)
        out_csv = data_dir / "processed_combustion_data.csv"
        raw.to_csv(out_csv, index=False)
        print_status(f"Consolidado guardado: {out_csv}")
    else:
        print_status("No se encontraron archivos de datos. Generando ejemplo sintético.")
        np.random.seed(7)
        n = 600
        t = pd.date_range("2025-01-01 12:00:00", periods=n, freq="S")
        # Perfíl simple representativo para demostración
        O2 = 5 + 1.0*np.sin(np.linspace(0, 6*np.pi, n)) + 0.2*np.random.randn(n)
        CO2 = 10 + 0.8*np.cos(np.linspace(0, 6*np.pi, n)) + 0.2*np.random.randn(n)
        CO = 20 + 100*np.exp(-0.5*(np.linspace(-2,2,n))**2) + 5*np.random.randn(n)  # ppm con pico
        NOx = 50 + 40*np.exp(-0.5*(np.linspace(1.2,-1.2,n))**2) + 5*np.random.randn(n)  # ppm con pico desplazado
        T = 200 + 20*np.sin(np.linspace(0, 4*np.pi, n)) + 2*np.random.randn(n)  # °C
        P = 101.3 + 0.1*np.random.randn(n)  # kPa

        raw = pd.DataFrame({
            "time": t,
            "O2": O2.clip(0, 20),
            "CO2": CO2.clip(0, 20),
            "CO": CO.clip(0),
            "NOx": NOx.clip(0),
            "T": T,
            "P": P,
            "source_file": "synthetic_example.csv",
        })
        out_csv = data_dir / "synthetic_example_combustion.csv"
        raw.to_csv(out_csv, index=False)
        print_status(f"Datos sintéticos guardados: {out_csv}")
    
    return raw


# %% [calc] Funciones de estequiometría y métricas

AIR_O2_FRAC = 0.21
AIR_N2_PER_O2 = 3.7619
MW_N2 = 28.97
MW_O2 = 32.00

@dataclass
class Fuel:
    x: float  # C
    y: float  # H
    name: str = "CxHy"

    @property
    def mw(self) -> float:
        return 12.011*self.x + 1.008*self.y

    @property
    def nu_O2(self) -> float:
        return self.x + self.y/4.0

    @property
    def afr_st(self) -> float:
        mass_air_per_mol_O2 = MW_O2 + AIR_N2_PER_O2*MW_N2
        return (self.nu_O2 * mass_air_per_mol_O2) / self.mw

    @property
    def co2_st_dry_frac(self) -> float:
        # fracción molar seca de CO2 a esteq.
        dry_moles = self.x + AIR_N2_PER_O2*self.nu_O2
        return (self.x / dry_moles)


def lambda_from_o2_co2(o2_pct: pd.Series, co2_pct: pd.Series, fuel: Fuel) -> pd.Series:
    """
    Cálculo de λ (exceso de aire) desde gases secos.
    λ ≈ (CO2_s,sec / CO2_med) · (21 / (21 − O2_med))
    """
    eps = 1e-9
    co2s = 100.0 * fuel.co2_st_dry_frac
    num = (co2s / (co2_pct.clip(lower=eps))) * (21.0 / (21.0 - o2_pct.clip(upper=20.999)))
    return num


def compute_combustion_metrics(df: pd.DataFrame, fuel: Fuel) -> pd.DataFrame:
    """Calcula métricas de combustión y añade suavizado."""
    out = df.copy()
    # Sanitizar - convertir a numérico con coerción de errores
    for col in ["O2", "CO2", "CO", "NOx", "T", "P"]:
        if col in out: 
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["fuel_name"] = fuel.name
    out["AFR_st"] = fuel.afr_st
    out["CO2_st_dry_frac"] = fuel.co2_st_dry_frac
    out["CO2_st_dry_pct"] = 100.0 * fuel.co2_st_dry_frac

    if set(["O2", "CO2"]).issubset(out.columns):
        out["lambda"] = lambda_from_o2_co2(out["O2"], out["CO2"], fuel)
        out["phi"] = 1.0 / out["lambda"].replace(0, np.nan)
    else:
        out["lambda"] = np.nan
        out["phi"] = np.nan

    # Suavizado opcional para señales ruidosas
    for col in ["CO", "NOx", "T"]:
        if col in out.columns and len(out) >= 21:
            try:
                # Interpolar NaNs antes de filtrar
                s = out[col].interpolate(limit_direction='both')
                out[f"{col}_sg"] = savgol_filter(s, window_length=21, polyorder=2)
            except Exception as e:
                print_status(f"Error al aplicar Savitzky-Golay a {col}: {e}")
                pass

    out_csv = DATA_DIR / "combustion_metrics.csv"
    out.to_csv(out_csv, index=False)
    print_status(f"Métricas guardadas: {out_csv}")
    return out


# %% [calc] Funciones de Temperatura adiabática simplificada

# Propiedades de combustibles comunes (LHV a 25°C)
FUEL_PROPERTIES = {
    "Metano CH4": {"x": 1, "y": 4, "LHV_kJ_mol": 802.0},
    "Propano C3H8": {"x": 3, "y": 8, "LHV_kJ_mol": 2043.0},
    "Octano C8H18": {"x": 8, "y": 18, "LHV_kJ_mol": 5074.0},
}

# Capacidades caloríficas promedio (J/mol·K) a ~1500 K
CP_PRODUCTS = {
    "CO2": 55.0,
    "H2O": 41.0,
    "N2": 32.0,
}

def estimate_adiabatic_temp(fuel: Fuel, lhv_kj_mol: float, lambda_ratio: float = 1.0, T0_K: float = 298.15) -> float:
    """
    Estimación simplificada de temperatura adiabática de llama.
    T_ad ≈ T_0 + LHV / (Σ n_prod·Cp_prod)
    """
    # Moles de productos por mol de combustible
    n_CO2 = fuel.x
    n_H2O = fuel.y / 2.0
    n_O2_req = fuel.nu_O2
    n_N2 = AIR_N2_PER_O2 * n_O2_req * lambda_ratio
    n_O2_excess = n_O2_req * (lambda_ratio - 1.0) if lambda_ratio > 1.0 else 0.0
    
    # Capacidad calorífica total de productos
    Cp_total = (n_CO2 * CP_PRODUCTS["CO2"] + 
                n_H2O * CP_PRODUCTS["H2O"] + 
                n_N2 * CP_PRODUCTS["N2"] + 
                n_O2_excess * 32.0)  # Cp de O2 exceso similar a N2
    
    # Energía liberada (LHV) / Cp total → ΔT
    delta_T = (lhv_kj_mol * 1000.0) / Cp_total  # Convertir kJ a J
    T_ad = T0_K + delta_T
    
    return T_ad

def calculate_adiabatic_temps(fuel: Fuel):
    """Calcula y guarda T_ad para varios lambdas."""
    fuel_name = fuel.name
    if fuel_name in FUEL_PROPERTIES:
        lhv = FUEL_PROPERTIES[fuel_name]["LHV_kJ_mol"]
    else:
        # Por defecto usar propano si no está en la tabla
        lhv = FUEL_PROPERTIES["Propano C3H8"]["LHV_kJ_mol"]
        print_status(f"Usando LHV de Propano como default: {lhv} kJ/mol")

    # Calcular para lambda = 1.0, 1.1, 1.2, 1.3
    lambda_values = [1.0, 1.1, 1.2, 1.3]
    T_ad_results = []

    for lam in lambda_values:
        T_ad_K = estimate_adiabatic_temp(fuel, lhv, lambda_ratio=lam)
        T_ad_C = T_ad_K - 273.15
        T_ad_results.append({
            "lambda": lam,
            "T_ad_K": T_ad_K,
            "T_ad_C": T_ad_C,
            "fuel": fuel_name,
            "LHV_kJ_mol": lhv
        })

    T_ad_df = pd.DataFrame(T_ad_results)
    out_csv = DATA_DIR / "temperatura_adiabatica.csv"
    T_ad_df.to_csv(out_csv, index=False)
    print_status(f"Temperaturas adiabáticas guardadas: {out_csv}")

    print(f"\nTemperatura adiabática estimada para {fuel_name}:")
    print(T_ad_df[["lambda", "T_ad_K", "T_ad_C"]].to_string(index=False))
    print("\nNota: Estimación simplificada con Cp promedio.")
    return T_ad_df


# %% [analysis] Funciones de Estadísticos descriptivos y outliers

def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Identifica outliers usando el método IQR."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k*iqr
    high = q3 + k*iqr
    return (s < low) | (s > high)

def run_descriptive_analysis(metrics: pd.DataFrame):
    """Calcula y guarda estadísticos descriptivos, outliers y móviles."""
    num_cols = [c for c in metrics.columns if metrics[c].dtype.kind in "fi" and c not in ["AFR_st", "CO2_st_dry_frac", "CO2_st_dry_pct"]]

    desc = metrics[num_cols].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
    out_csv = DATA_DIR / "stats_descriptivos.csv"
    desc.to_csv(out_csv)
    print_status(f"Estadísticos guardados: {out_csv}")

    outlier_flags = pd.DataFrame(index=metrics.index)
    for c in num_cols:
        outlier_flags[c] = iqr_outliers(metrics[c].dropna())

    outlier_rate = outlier_flags.mean().sort_values(ascending=False)
    out_csv = DATA_DIR / "stats_outliers_iqr.csv"
    outlier_rate.to_csv(out_csv, header=["outlier_rate"])
    print_status(f"Outliers IQR guardados: {out_csv}")
    
    print_status("Tasas de outliers (0.0 = sin outliers):")
    if outlier_rate.max() == 0:
        print_status("No se detectaron outliers con el método IQR (k=1.5).")
    else:
        # Imprimir solo las columnas que SÍ tienen outliers
        print(outlier_rate[outlier_rate > 0].to_string())

    # Rolling stats (ventana ~ 30s si hay muestreo 1 Hz)
    if len(metrics) >= 30:
        roll = metrics[num_cols].rolling(30, min_periods=10).agg(["mean", "std"]).copy()
        out_csv = DATA_DIR / "stats_rolling_30.csv"
        roll.to_csv(out_csv)
        print_status(f"Estadísticos móviles guardados: {out_csv}")

    # Máximos y mínimos de interés
    peaks_summary = {
        "O2_max": float(metrics.get("O2", pd.Series([np.nan])).max()),
        "CO2_max": float(metrics.get("CO2", pd.Series([np.nan])).max()),
        "CO_max_ppm": float(metrics.get("CO", pd.Series([np.nan])).max()),
        "NOx_max_ppm": float(metrics.get("NOx", pd.Series([np.nan])).max()),
        "T_max_C": float(metrics.get("T", pd.Series([np.nan])).max()),
        "lambda_min": float(metrics.get("lambda", pd.Series([np.nan])).min()),
    }
    with open(DATA_DIR / "summary_maximos.json", "w", encoding="utf-8") as f:
        json.dump(peaks_summary, f, ensure_ascii=False, indent=2)
    print_status("Resumen de máximos guardado: summary_maximos.json")


# %% [analysis] Función de Detección de picos en señales (VERSIÓN CORREGIDA)
def find_and_save_peaks(metrics: pd.DataFrame):
    """
    Encuentra picos en CO, NOx y T y los guarda en CSVs.
    Utiliza percentiles y prominencia para una detección robusta.
    """
    peaks_tables = {}
    for col in ["CO", "NOx", "T"]:
        if col in metrics.columns:
            s = metrics[col].astype(float).ffill().bfill()
            if s.isnull().all():
                print_status(f"Columna {col} está vacía, saltando detección de picos.")
                continue
            
            # --- INICIO BLOQUE MODIFICADO (Lógica Robusta de Picos) ---
            
            # 1. Altura mínima: El pico debe estar al menos en el percentil 90
            height_threshold = s.quantile(0.90) 
            
            # 2. Prominencia: El pico debe "sobresalir" al menos un 5% del rango total
            s_max = s.max()
            s_min = s.min()
            # Añadir un valor pequeño (epsilon) para evitar prominencia 0 si todos los datos son iguales
            epsilon = 1e-9
            prominence_threshold = (s_max - s_min) * 0.05 + epsilon
            
            # 3. Distancia: separación mínima entre picos
            distance = max(5, len(s)//100)
            
            idx, props = find_peaks(
                s.values, 
                height=height_threshold, 
                prominence=prominence_threshold,
                distance=distance
            )
            # --- FIN BLOQUE MODIFICADO ---
            
            if len(idx) == 0:
                print_status(f"No se encontraron picos para {col} con los nuevos umbrales (P90 + 5% prominencia).")
                peaks_df = pd.DataFrame(columns=["index", "time", col]) # Guardar DF vacío
            else:
                print_status(f"Detectados {len(idx)} picos para {col}.")
                time_values = metrics.loc[idx, "time"].values if "time" in metrics.columns and not metrics.loc[idx, "time"].empty else idx
                
                peaks_df = pd.DataFrame({
                    "index": idx,
                    "time": time_values,
                    col: s.iloc[idx].values,
                })
            
            peaks_tables[col] = peaks_df
            out_csv = DATA_DIR / f"peaks_{col}.csv" # Nombre de archivo final
            peaks_df.to_csv(out_csv, index=False)
            print_status(f"Picos de {col} guardados: {out_csv}")
    
    return peaks_tables


# %% [viz] Funciones de Visualización

def plot_time_series(metrics: pd.DataFrame):
    """Grafica series temporales de O2, CO2, CO, NOx y lambda."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Usar 'time' si es datetime, si no, usar el índice
    if pd.api.types.is_datetime64_any_dtype(metrics.get("time")):
        x = metrics["time"]
        xlabel = "Tiempo"
    else:
        x = metrics.index
        xlabel = "Índice de muestra"

    if "O2" in metrics.columns: axes[0].plot(x, metrics["O2"], label="O2 [%]")
    if "CO2" in metrics.columns: axes[0].plot(x, metrics["CO2"], label="CO2 [%]")
    axes[0].set_ylabel("[%]")
    axes[0].legend(loc="best")
    axes[0].set_title("Gases principales (Base Seca)")

    if "CO" in metrics.columns: axes[1].plot(x, metrics["CO"], label="CO [ppm]")
    if "NOx" in metrics.columns: axes[1].plot(x, metrics["NOx"], label="NOx [ppm]")
    axes[1].set_ylabel("[ppm]")
    axes[1].legend(loc="best")
    axes[1].set_title("Emisiones (ppm)")

    if "lambda" in metrics.columns: axes[2].plot(x, metrics["lambda"], label="λ")
    axes[2].axhline(1.0, color="k", ls="--", lw=1, label="λ = 1.0")
    axes[2].set_ylabel("λ [-]")
    axes[2].set_xlabel(xlabel)
    axes[2].legend(loc="best")
    axes[2].set_title("Exceso de Aire (λ)")

    save_fig(fig, "series_temporales")
    plt.close(fig)

def plot_hist_corr(metrics: pd.DataFrame):
    """Grafica histogramas y matriz de correlación."""
    plot_cols = [c for c in ["O2", "CO2", "CO", "NOx", "T", "lambda"] if c in metrics.columns]
    if not plot_cols:
        return

    # Histogramas
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(4*len(plot_cols), 3))
    if len(plot_cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, plot_cols):
        sns.histplot(metrics[c].dropna(), bins=30, ax=ax, kde=True)
        ax.set_title(c)
    save_fig(fig, "histogramas")
    plt.close(fig)

    # Matriz de correlación
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = metrics[plot_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlación")
    save_fig(fig, "correlacion")
    plt.close(fig)

def plot_boxplots(metrics: pd.DataFrame):
    """Grafica box plots para distribuciones y outliers."""
    viz_cols = [c for c in ["O2", "CO2", "CO", "NOx", "T", "lambda", "phi"] if c in metrics.columns]
    if len(viz_cols) < 2:
        return
        
    fig, axes = plt.subplots(1, len(viz_cols), figsize=(3*len(viz_cols), 4))
    if len(viz_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, viz_cols):
        data = metrics[col].dropna()
        if not data.empty:
            ax.boxplot(data, vert=True)
        ax.set_ylabel(col)
        ax.set_title(f"Box plot: {col}")
        ax.grid(True, alpha=0.3)
    
    save_fig(fig, "boxplots")
    plt.close(fig)

def plot_violin(metrics: pd.DataFrame):
    """Grafica violin plots para densidad de distribuciones."""
    viz_cols = [c for c in ["O2", "CO2", "CO", "NOx", "T", "lambda", "phi"] if c in metrics.columns]
    if len(viz_cols) < 2:
        return
        
    # Preparar datos en formato largo para seaborn
    plot_data = metrics[viz_cols].copy()
    plot_data_long = plot_data.melt(var_name="Variable", value_name="Valor")
    
    fig, ax = plt.subplots(figsize=(max(8, len(viz_cols)*1.2), 5))
    sns.violinplot(data=plot_data_long, x="Variable", y="Valor", hue="Variable", ax=ax, palette="Set2", legend=False)
    ax.set_title("Violin plots - Distribución de variables")
    ax.set_ylabel("Valor")
    ax.tick_params(axis='x', rotation=45)
    
    save_fig(fig, "violinplots")
    plt.close(fig)

def plot_pairplot(metrics: pd.DataFrame):
    """Grafica scatter matrix (pairplot) para relaciones bivariadas."""
    pair_cols = [c for c in ["O2", "CO2", "CO", "NOx", "lambda"] if c in metrics.columns]
    if len(pair_cols) < 2:
        return

    plot_sample = metrics[pair_cols].dropna()
    
    # Si hay muchos datos, submuestrear para velocidad
    if len(plot_sample) > 1000:
        plot_sample = plot_sample.sample(n=1000, random_state=42)
    
    if plot_sample.empty:
        print_status("No hay datos suficientes para el pairplot.")
        return

    # Crear pairplot con seaborn
    g = sns.pairplot(plot_sample, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 20}, corner=True)
    g.fig.suptitle("Scatter Matrix - Relaciones bivariadas", y=1.01)
    
    save_fig(g.fig, "pairplot", tight=False)
    plt.close(g.fig)

def plot_quartiles(metrics: pd.DataFrame):
    """Grafica distribuciones por cuartiles e IQR."""
    viz_cols = [c for c in ["O2", "CO2", "CO", "NOx", "T", "lambda", "phi"] if c in metrics.columns]
    if len(viz_cols) < 3:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Cuartiles de las variables principales
    quartile_data = {}
    plot_cols = [col for col in viz_cols if col in metrics.columns and not metrics[col].dropna().empty]
    
    for col in plot_cols[:4]:  # Limitar a 4 variables principales
        q = metrics[col].quantile([0.25, 0.5, 0.75])
        quartile_data[col] = [q[0.25], q[0.5], q[0.75]]
    
    if quartile_data:
        df_q = pd.DataFrame(quartile_data, index=["Q1", "Q2 (mediana)", "Q3"]).T
        
        # Gráfico de barras agrupadas
        df_q.plot(kind="bar", ax=axes[0], width=0.8)
        axes[0].set_title("Cuartiles por variable")
        axes[0].set_ylabel("Valor")
        axes[0].legend(title="Cuartil")
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de rango intercuartílico (IQR)
        iqr_vals = [df_q.loc[col, "Q3"] - df_q.loc[col, "Q1"] for col in df_q.index]
        axes[1].bar(df_q.index, iqr_vals, color="steelblue", alpha=0.7)
        axes[1].set_title("Rango Intercuartílico (IQR) por variable")
        axes[1].set_ylabel("IQR")
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        save_fig(fig, "cuartiles_iqr")
    
    plt.close(fig)

def plot_lambda_temp(metrics: pd.DataFrame):
    """Grafica lambda vs temperatura con scatter density."""
    if "lambda" not in metrics.columns or "T" not in metrics.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    lam_data = metrics["lambda"].dropna()
    T_data = metrics["T"].dropna()
    
    # Asegurar mismos índices
    common_idx = lam_data.index.intersection(T_data.index)
    lam_plot = lam_data.loc[common_idx]
    T_plot = T_data.loc[common_idx]

    if lam_plot.empty or T_plot.empty:
        print_status("No hay datos comunes de Lambda y T para graficar.")
        plt.close(fig)
        return
    
    scatter = ax.scatter(lam_plot, T_plot, c=T_plot, cmap="coolwarm", 
                        alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
    ax.axvline(1.0, color="green", ls="--", lw=2, label="λ = 1 (estequiométrico)")
    ax.set_xlabel("λ (Exceso de aire)")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title("Relación λ vs Temperatura")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Temperatura [°C]")
    
    save_fig(fig, "lambda_vs_temperatura")
    plt.close(fig)


# %% [export] Función para el Resumen de archivos generados (CORREGIDA)
def generate_manifest():
    """Genera un JSON manifest de todos los archivos de salida."""
    outputs = {
        "tables": [
            str(DATA_DIR / "processed_combustion_data.csv"),
            str(DATA_DIR / "synthetic_example_combustion.csv"),
            str(DATA_DIR / "combustion_metrics.csv"),
            str(DATA_DIR / "temperatura_adiabatica.csv"),
            str(DATA_DIR / "stats_descriptivos.csv"), # <-- CORREGIDO
            str(DATA_DIR / "stats_outliers_iqr.csv"),
            str(DATA_DIR / "stats_rolling_30.csv"),
            str(DATA_DIR / "peaks_CO.csv"),
            str(DATA_DIR / "peaks_NOx.csv"),
            str(DATA_DIR / "peaks_T.csv"),
            str(DATA_DIR / "summary_maximos.json"),
        ],
        "figures": [
            str(DATA_DIR / "figures" / "series_temporales.png"),
            str(DATA_DIR / "figures" / "series_temporales.svg"),
            str(DATA_DIR / "figures" / "histogramas.png"),
            str(DATA_DIR / "figures" / "histogramas.svg"),
            str(DATA_DIR / "figures" / "correlacion.png"),
            str(DATA_DIR / "figures" / "correlacion.svg"),
            str(DATA_DIR / "figures" / "boxplots.png"),
            str(DATA_DIR / "figures" / "boxplots.svg"),
            str(DATA_DIR / "figures" / "violinplots.png"),
            str(DATA_DIR / "figures" / "violinplots.svg"),
            str(DATA_DIR / "figures" / "pairplot.png"),
            str(DATA_DIR / "figures" / "pairplot.svg"),
            str(DATA_DIR / "figures" / "cuartiles_iqr.png"),
            str(DATA_DIR / "figures" / "cuartiles_iqr.svg"),
            str(DATA_DIR / "figures" / "lambda_vs_temperatura.png"),
            str(DATA_DIR / "figures" / "lambda_vs_temperatura.svg"),
        ],
    }
    # Filtrar solo los archivos que realmente existen
    outputs["tables"] = [f for f in outputs["tables"] if Path(f).exists()]
    outputs["figures"] = [f for f in outputs["figures"] if Path(f).exists()]

    with open(DATA_DIR / "outputs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print_status("Manifest de salidas: outputs_manifest.json")
    print(f"\nTotal tablas generadas: {len(outputs['tables'])}")
    print(f"Total figuras generadas: {len(outputs['figures'])} ({len(outputs['figures'])//2} gráficos en PNG+SVG)")
    return outputs


# %% Función principal
def main():
    """Ejecuta el flujo completo de análisis de combustión."""
    
    # Asegurarse de que el directorio de datos existe
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print_status(f"Directorio de datos (entrada y salida): {DATA_DIR}")

    # --- Carga y preprocesado ---
    raw_data = load_data(DATA_DIR)
    if raw_data.empty:
        print_status("No se cargaron datos. Terminando script.")
        return
        
    print("\n--- Vista de datos crudos (primeras 3 filas) ---")
    print(raw_data.head(3))
    print("-" * 50)

    # --- Cálculos Estequiométricos ---
    # Configura aquí el combustible de interés (ejemplos comunes)
    # Metano CH4: x=1, y=4 ; Propano C3H8: x=3, y=8 ; Gasolina aprox.: C8H18
    FUEL = Fuel(x=3, y=8, name="Propano C3H8")
    
    metrics_df = compute_combustion_metrics(raw_data, FUEL)
    print("\n--- Vista de datos con métricas (primeras 3 filas) ---")
    print(metrics_df.head(3))
    print("-" * 50)
    
    # --- Cálculo de Temperatura Adiabática ---
    _ = calculate_adiabatic_temps(FUEL)
    print("-" * 50)

    # --- Análisis Descriptivo ---
    print_status("Iniciando análisis descriptivo...")
    run_descriptive_analysis(metrics_df)
    print("-" * 50)

    # --- Detección de Picos ---
    print_status("Iniciando detección de picos (lógica robusta)...")
    _ = find_and_save_peaks(metrics_df) 
    print("-" * 50)

    # --- Visualizaciones ---
    print_status("Generando visualizaciones...")
    plot_time_series(metrics_df)
    plot_hist_corr(metrics_df)
    plot_boxplots(metrics_df)
    plot_violin(metrics_df)
    plot_pairplot(metrics_df)
    plot_quartiles(metrics_df)
    plot_lambda_temp(metrics_df)
    print_status("Visualizaciones guardadas.")
    print("-" * 50)

    # --- Reporte Final ---
    print_status("Generando manifiesto de salidas...")
    _ = generate_manifest()
    print_status("Análisis completado.")

# %% Punto de entrada del script
if __name__ == "__main__":
    main()