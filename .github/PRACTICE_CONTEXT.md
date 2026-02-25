# Practice Structure & Tooling Context

> **Purpose:** Reference document defining the canonical structure, conventions,
> toolchain, and patterns used in **P01 (Protección Láser)** — the template
> practice. Every new practice (P02–P09) must follow these rules to ensure
> complete uniformity across manuals, presentations, notebooks, and scripts.

---

## 1. Repository Layout

```
Aero-Messure/
├── .github/
│   └── PRACTICE_CONTEXT.md           ← THIS FILE
├── manuals/
│   ├── BUILD.md                      ← LaTeX build instructions
│   └── PXX_<Name>/
│       ├── manual_PXX.tex            ← Lab manual (article, XeLaTeX)
│       ├── references_PXX.bib        ← BibLaTeX bibliography (IEEE style)
│       └── presentation/
│           └── presentacion_PXX.tex  ← Beamer presentation (XeLaTeX)
├── Practicas/
│   ├── README.md                     ← Methodology summaries for all practices
│   └── PXX_<Name>/
│       ├── README.md                 ← Practice-specific checklist
│       ├── data/
│       │   ├── .gitkeep
│       │   ├── *.csv / *.txt         ← Input datasets
│       │   └── images/               ← Generated plots (by script/notebook)
│       ├── notebooks/
│       │   └── PXX_<Name>.ipynb      ← Jupyter notebook (full analysis)
│       └── src/
│           └── PXX_<Name>.py         ← Standalone Python script (.py mirror)
├── files/                            ← Legacy/raw experimental data
├── requirements.txt                  ← Python dependencies (project-wide)
└── README.md                         ← Project overview
```

### Naming conventions

| Item | Pattern | Example |
|------|---------|---------|
| Practice folder | `PXX_<ShortName>` | `P01_Proteccion_Laser` |
| Manual TeX | `manual_PXX.tex` | `manual_P01.tex` |
| Bibliography | `references_PXX.bib` | `references_P01.bib` |
| Presentation TeX | `presentacion_PXX.tex` | `presentacion_P01.tex` |
| Notebook | `PXX_<Name>.ipynb` | `P01_Proteccion_Laser.ipynb` |
| Python script | `PXX_<Name>.py` | `P01_Proteccion_Laser.py` |
| Practice README | `Practicas/PXX_<Name>/README.md` | |

---

## 2. LaTeX Toolchain

### 2.1 Engine & Build

| Tool | Version | Purpose |
|------|---------|---------|
| **XeLaTeX** | ≥ 4.x (MiKTeX 25+) | Main engine — UTF-8 native, `fontspec` |
| **Biber** | ≥ 2.17 | Bibliography backend for `biblatex` |
| **Latexmk** | ≥ 4.80 | Build automation |

```powershell
# Manual
cd manuals\PXX_<Name>
latexmk -xelatex -interaction=nonstopmode manual_PXX.tex

# Presentation
cd manuals\PXX_<Name>\presentation
latexmk -xelatex -interaction=nonstopmode presentacion_PXX.tex
```

> **Exit code 1** is normal (warnings only, not fatal errors).

### 2.2 Manual Preamble (canonical)

```latex
\documentclass[12pt,a4paper]{article}

% ---- Encoding & Language ----
\usepackage{fontspec}                         % XeLaTeX — replaces inputenc/fontenc
\usepackage[spanish]{babel}
\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}  % "Tabla" not "Cuadro"

% ---- Page Layout ----
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\setlength{\headheight}{14.5pt}

% ---- Math & Physics ----
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{mathtools}
\usepackage{siunitx}
\sisetup{output-decimal-marker={,}, group-separator={.}}

% ---- Graphics & Color ----
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

% ---- TikZ ----
\usepackage{tikz}
\usetikzlibrary{babel}                       % MUST be first — fixes spanish babel + arrows
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,
                decorations.pathmorphing,decorations.markings,patterns,
                fit,backgrounds}

% ---- Tables ----
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{colortbl}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{longtable}

% ---- Code Listings ----
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue!70!black}\bfseries,
  commentstyle=\color{green!50!black},
  stringstyle=\color{red!60!black},
  backgroundcolor=\color{gray!10},
  frame=single, breaklines=true,
  numbers=left, numberstyle=\tiny\color{gray}
}

% ---- Hyperlinks ----
\usepackage[colorlinks=true,linkcolor=blue!70!black,
            citecolor=green!50!black,urlcolor=blue!50!black]{hyperref}

% ---- Bibliography ----
\usepackage[backend=biber,style=ieee,sorting=none]{biblatex}
\addbibresource{references_PXX.bib}

% ---- Headers & Footers ----
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Técnicas de Medida -- Ingeniería Aeronáutica}
\fancyhead[R]{\small PXX: <Short Title>}
\fancyfoot[C]{\thepage}

% ---- Custom Colors (shared palette) ----
\definecolor{aeroblue}{RGB}{0,51,102}
\definecolor{aerored}{RGB}{153,0,0}
\definecolor{aerogreen}{RGB}{0,102,51}
\definecolor{aeroyellow}{RGB}{255,204,0}
```

### 2.3 Presentation Preamble (canonical)

```latex
\documentclass[aspectratio=169,12pt]{beamer}

\usetheme{Madrid}
\usecolortheme{seahorse}
\usefonttheme{professionalfonts}

% ---- Encoding & Language ----
\usepackage{fontspec}
\usepackage[spanish]{babel}
\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}

% ---- Math & Physics ----
\usepackage{amsmath,amssymb}
\usepackage{mathtools}
\usepackage{siunitx}
\sisetup{output-decimal-marker={,}}

% ---- TikZ ----
\usepackage{tikz}
\usetikzlibrary{babel}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,decorations.markings}

% ---- Tables ----
\usepackage{booktabs}
\usepackage{colortbl}
\usepackage{tabularx}

% ---- Code ----
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\tiny,
  keywordstyle=\color{blue!80}\bfseries,
  commentstyle=\color{green!50!black},
  backgroundcolor=\color{gray!10},
  frame=single, breaklines=true
}

% ---- Custom Colors (shared palette) ----
\definecolor{aeroblue}{RGB}{0,51,102}
\definecolor{aerored}{RGB}{153,0,0}
\definecolor{aerogreen}{RGB}{0,102,51}
\definecolor{aeroyellow}{RGB}{204,153,0}
\definecolor{aerogreendark}{RGB}{0,80,40}
\definecolor{aerogreenlight}{RGB}{0,140,70}

% ---- GREEN THEME ----
\setbeamercolor{frametitle}{bg=aerogreen,fg=white}
\setbeamercolor{title}{bg=aerogreen,fg=white}
\setbeamercolor{structure}{fg=aerogreen}
\setbeamercolor{palette primary}{bg=aerogreen,fg=white}
\setbeamercolor{palette secondary}{bg=aerogreenlight,fg=white}
\setbeamercolor{palette tertiary}{bg=aerogreendark,fg=white}
\setbeamercolor{palette quaternary}{bg=aerogreendark,fg=white}
\setbeamercolor{section in toc}{fg=aerogreen}
\setbeamercolor{subsection in toc}{fg=aerogreendark}
\setbeamercolor{block title}{bg=aerogreen,fg=white}
\setbeamercolor{block body}{bg=aerogreen!10,fg=black}
\setbeamercolor{block title alerted}{bg=aerored,fg=white}
\setbeamercolor{block body alerted}{bg=aerored!10,fg=black}
\setbeamercolor{block title example}{bg=aerogreenlight,fg=white}
\setbeamercolor{block body example}{bg=aerogreenlight!10,fg=black}
\setbeamercolor{item}{fg=aerogreen}
\setbeamercolor{itemize item}{fg=aerogreen}
\setbeamercolor{enumerate item}{fg=aerogreen}

% ---- Title Info ----
\title[PXX -- Short]{PXX: Full Title}
\subtitle{Técnicas de Medida -- Ingeniería Aeronáutica}
\author{Laboratorio de Técnicas de Medida}
\date{}                                       % NO date — intentionally blank
```

### 2.4 Key LaTeX Rules

| Rule | Detail |
|------|--------|
| **Engine** | Always XeLaTeX (`fontspec`), never pdfLaTeX (`inputenc`/`fontenc`) |
| **Table captions** | Must say "Tabla", not "Cuadro" — enforced by `\addto\captionsspanish` |
| **TikZ + babel** | `\usetikzlibrary{babel}` MUST be loaded before other tikz libraries |
| **S-column literals** | Wrap non-numeric content in braces: `{--}` inside `S` columns |
| **Title page** | No `\date{}` — leave blank; no department name |
| **Header** | Left: "Técnicas de Medida -- Ingeniería Aeronáutica"; Right: "PXX: Title" |
| **Bibliography** | IEEE style via `biblatex` + `biber`; file = `references_PXX.bib` |
| **siunitx** | Decimal marker = comma, group separator = period (Spanish locale) |
| **Color palette** | `aeroblue`, `aerored`, `aerogreen`, `aeroyellow` — exact RGB values above |

---

## 3. Manual Content Structure (Section Outline)

Every manual must follow this section order:

```
1.  Introducción
2.  Objetivos de Aprendizaje
3.  Fundamentos Teóricos
    3.x Subsections as needed by topic
4.  Escenarios / Datos de la Práctica
    4.1 Sources / Equipment description
    4.2 Data catalogue (tables)
    4.3 Reference values (MPE, calibration curves, etc.)
5.  Metodología de Cálculo
    5.x Step-by-step derivation (numbered subsections)
6.  Ejemplo Resuelto: Escenarios Base
    6.x One subsection per solved scenario
    6.last Tabla Resumen (summary table of all base results)
    6.last+1 Evaluación de Riesgo de los Escenarios Base
7.  Casos de Estudio por Equipos
    - Overview: general instructions, task enumeration, risk template table
    7.x One subsection per team (Equipo 1..4)
        - Parameters, tasks, risk evaluation guidance, 3 additional problems
8.  Evaluación de Riesgos
    8.1 Matriz de Riesgos (5×5 grid: P×S=R, Bajo/Medio/Alto)
    8.2 Jerarquía de Controles
9.  Herramientas de Software
    9.1 Notebook Jupyter
    9.2 Script Python
    9.3 Datos
10. Procedimiento Experimental
11. Cuestionario (10 questions, no solutions)
12. Formato del Informe
13. Conclusiones
Appendix: Reference tables (MPE, etc.)
Bibliography (via \printbibliography)
```

### Risk evaluation pattern

Every practice that involves hazardous equipment must include:

1. **Base scenario risk table** (after the solved examples summary):

   | Escenario | P | S | R | Nivel | Controles recomendados |
   |-----------|---|---|---|-------|------------------------|
   | ...       | 1-5 | 1-5 | P×S | Bajo/Medio/Alto | Text |

2. **Risk template table** (blank, for students to fill):

   | Escenario | Probabilidad (1-5) | Severidad (1-5) | R = P×S | Nivel | Controles propuestos | Posición en matriz |
   |-----------|---------------------|------------------|---------|-------|----------------------|--------------------|
   | (empty)   | | | | | | |

3. **Per-team risk guidance** in each team subsection.

4. **Risk classification levels:**
   - **Bajo** (green): R < 5
   - **Medio** (yellow): 5 ≤ R ≤ 12
   - **Alto** (red): R > 12

5. **Probability scale:** 1=MuyBajo, 2=Bajo, 3=Medio, 4=Alto, 5=MuyAlto
6. **Severity scale:** 1=Insignificante, 2=Menor, 3=Moderado, 4=Mayor, 5=Catastrófico

---

## 4. Presentation Content Structure (Slide Outline)

```
Section 1: Introducción
  - Title slide (auto)
  - Table of contents
  - Objectives slide

Section 2: Fundamentos Teóricos
  - Key concepts (3-5 slides)
  - Diagrams (TikZ)
  - Classification tables

Section 3: Escenarios
  - Equipment/scenarios overview
  - Data catalogue (compact tables)
  - Evaluación de Riesgo — Escenarios Base (risk table slide)

Section 4: Cálculos de Protección / Metodología
  - Step-by-step formulas
  - Solved examples (1 slide per scenario)
  - Summary table

Section 5: Casos de Estudio por Equipos
  - Overview slide: task list + risk evaluation instructions
  - 1 slide per team: 3-column layout
    Column 1: \begin{tabular} with parameters
    Column 2: \begin{alertblock}{Riesgo} with risk hints
    Column 3: \begin{exampleblock}{Problemas} with problem list

Section 6: Controles de Seguridad / Supplementary
  - Control hierarchy, 5×5 risk matrix (TikZ), etc.

Section 7: Herramientas de Software
  - Notebook description + code snippet
  - Script description + function list
  - Data files

Section 8: Cuestionario y Formato
  - Key questions (subset of manual's 10)
  - Report format requirements

Section 9: Conclusiones
  - 5-6 bullet points, includes risk evaluation mention
  - Final references slide
```

### Beamer conventions

| Rule | Detail |
|------|--------|
| **Theme** | Madrid + seahorse + green override (see preamble above) |
| **Aspect ratio** | 16:9 (`aspectratio=169`) |
| **Font size** | 12pt base |
| **Block types** | `block` (green), `alertblock` (red), `exampleblock` (light green) |
| **No date** | `\date{}` — left blank |
| **No institute** | Omit `\institute{}` (no department) |
| **Team slides** | Use tabular + alertblock + exampleblock (not plain itemize blocks) |

---

## 5. Python Script Structure

Each practice has a standalone `.py` in `Practicas/PXX_<Name>/src/`.

### 5.1 File layout

```python
#!/usr/bin/env python3
"""
# PXX: <Practice Title>

Script para <brief description>.
Este script es la versión .py equivalente del notebook PXX_<Name>.ipynb.

<Data sources and parameters>
"""

# --- IMPORTS ---
import math, csv, sys, json, os

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import ipywidgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

# --- CONSTANTS / LABELS ---
# (Practice-specific constants here)

# --- CORE FUNCTIONS ---
def load_data(path): ...
def compute_<metric>(...): ...
def evaluate_<scenario>(...): ...

# --- RISK FUNCTIONS (if applicable) ---
RISK_PROB_LABELS = {1:'MuyBajo', 2:'Bajo', 3:'Medio', 4:'Alto', 5:'MuyAlto'}
RISK_SEV_LABELS  = {1:'Insignificante', 2:'Menor', 3:'Moderado', 4:'Mayor', 5:'Catastrófico'}

def risk_assessment(prob: int, severity: int) -> dict: ...
def risk_controls(nivel: str) -> str: ...
def plot_risk_matrix(scenarios: list, save_path: str = None): ...

# --- PLOTTING / EVALUATION ---
def evaluate_and_plot(scenario, data, save_dir): ...

# --- MAIN ---
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', '<input_file>')
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Define scenarios (base + team)
    SCENARIOS = [...]
    TEAM_SCENARIOS_RISK = [...]

    # Load data
    data = load_data(DATA_PATH)

    # Evaluate base scenarios
    for s in SCENARIOS:
        evaluate_and_plot(s, data, IMAGE_DIR)
        # Risk assessment (if applicable)

    # Evaluate team risk scenarios
    for sc in TEAM_SCENARIOS_RISK:
        risk = risk_assessment(sc['prob'], sc['sev'])
        print(...)

    # Generate combined risk matrix
    plot_risk_matrix(all_scenarios, save_path=...)

    print("--- Análisis completo finalizado ---")

if __name__ == "__main__":
    main()
```

### 5.2 Rules for `.py` scripts

| Rule | Detail |
|------|--------|
| **Encoding** | UTF-8 source, but avoid non-ASCII in `print()` output (Windows cp1252 issue) |
| **Non-ASCII in output** | Use ASCII replacements: `->` not `→`, `-` not `–`, `CO2` not `CO₂` |
| **Non-ASCII in comments/docstrings** | OK — not printed to console |
| **Graceful imports** | `try/except` for `pandas`, `matplotlib`, `ipywidgets` |
| **Matplotlib backend** | `matplotlib.use('Agg')` before `import matplotlib.pyplot` |
| **Path handling** | Use `os.path` relative to `__file__`; create `data/images/` with `os.makedirs` |
| **Function naming** | snake_case, descriptive: `load_lenses()`, `risk_assessment()`, `evaluate_and_plot()` |
| **Data flow** | Load CSV → define scenarios → evaluate → plot → print summary |

---

## 6. Jupyter Notebook Structure

Each practice has a `.ipynb` in `Practicas/PXX_<Name>/notebooks/`.

### 6.1 Cell layout (canonical order)

| Cell # | Type | Purpose |
|--------|------|---------|
| 1 | Markdown | Title, objectives summary, practice description |
| 2 | Code | Imports + optional dependency checks |
| 3 | Code | Data loading (CSV) + preview |
| 4 | Code | Core functions: calculations, evaluation, risk functions |
| 5 | Code | Base scenario definitions + evaluation loop (with risk) |
| 6 | Code | Risk matrix visualization (all scenarios: base + team) |
| 7 | Code | Interactive widgets (ipywidgets) for custom scenarios |

### 6.2 Rules for notebooks

| Rule | Detail |
|------|--------|
| **Mirror the .py** | Notebook and script must contain the same logic/functions |
| **Pandas display** | Use `display(df)` in notebook, `.to_string()` in script |
| **Plots** | Show inline in notebook, save to `../data/images/` in script |
| **Risk visualization** | Dedicated cell for risk matrix (separate from base evaluation) |
| **Widgets** | Last cell; check `HAS_WIDGETS` before creating |
| **Relative paths** | Use `../data/` from notebooks directory |

---

## 7. Data Files

### 7.1 CSV format

- UTF-8 encoding, comma-separated
- First row = header with snake_case column names
- Example (P01 `epo_lenses.csv`):

```csv
lens_id,brand,model,band_lo_nm,band_hi_nm,od_value,vlt_pct
L-01,Thorlabs,LG1,190,540,5,40
```

### 7.2 Output images

- Saved to `Practicas/PXX_<Name>/data/images/`
- Format: PNG, 150 dpi
- Naming: `<scenario_name>.png`, `risk_matrix.png`
- Created by both the `.py` script and the notebook

---

## 8. Bibliography (BibLaTeX + Biber)

### 8.1 File: `references_PXX.bib`

- Use `@standard`, `@book`, `@article`, `@techreport` entry types
- Always include international standards (ISO, IEC, EN, ANSI) relevant to the practice
- Include Spanish regulations (Real Decreto) where applicable
- Example entries (P01): `iec60825`, `en207`, `ansi2014`, `svelto2010`, `saleh2007`, `rd486_2010`, `henderson2004`

### 8.2 Citation style

```latex
\usepackage[backend=biber,style=ieee,sorting=none]{biblatex}
\addbibresource{references_PXX.bib}
% ...
\printbibliography[heading=bibintoc,title={Referencias}]
```

---

## 9. Practice README Template

Each `Practicas/PXX_<Name>/README.md` should follow:

```markdown
# PXX - <Practice Title in Spanish>

Objetivo: <one-line objective>.

Checklist:
- <Step 1>
- <Step 2>
- <Step 3>
- Matriz de riesgos y controles.

Estructura:
- `data/`: <description of input data>.
- `src/`: <description of Python utilities>.
- `notebooks/`: <description of Jupyter analysis>.

Notebook principal:
- `notebooks/PXX_<Name>.ipynb` → incluye:
  - Resumen de objetivos y escenarios.
  - Cálculos principales.
  - Evaluación de riesgo con matriz 5×5.
  - Celda para escenarios personalizados.
```

---

## 10. Differences from Legacy (P02–P04 current state)

Practices P02–P04 were created before the P01 standardisation. They need these
updates to match the canonical template:

| Item | Legacy (P02-P04) | Canonical (P01) |
|------|-------------------|-----------------|
| Engine | pdfLaTeX (`inputenc`/`fontenc`) | XeLaTeX (`fontspec`) |
| Table caption | "Cuadro" (babel default) | "Tabla" (`\addto\captionsspanish`) |
| TikZ + babel | Missing `\usetikzlibrary{babel}` | Required (fixes `->` arrows) |
| Title date | `\date{\today}` | `\date{}` (blank) |
| Institute | `\institute{Departamento...}` | Omitted |
| Presentation theme color | Blue (`aeroblue`) | Green (`aerogreen` + full override) |
| Table packages | `booktabs` only | + `colortbl`, `tabularx`, `adjustbox`, `longtable` |
| Risk evaluation | Not present | Full 5×5 matrix (base + teams) |
| Team case studies | Varies | 4 teams, 3 problems each, no solutions, risk per team |
| Python risk functions | Not present | `risk_assessment()`, `risk_controls()`, `plot_risk_matrix()` |

### Migration checklist for each practice

- [ ] Replace `\usepackage[utf8]{inputenc}` + `\usepackage[T1]{fontenc}` with `\usepackage{fontspec}`
- [ ] Add `\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}`
- [ ] Add `\usetikzlibrary{babel}` before other tikz libraries
- [ ] Remove `\date{\today}`, use `\date{}`
- [ ] Remove `\institute{...}`
- [ ] Apply green theme override to presentation (copy block from §2.3)
- [ ] Add table packages: `colortbl`, `tabularx`, `adjustbox`, `longtable`
- [ ] Add risk evaluation section to manual (§8 in outline)
- [ ] Add risk evaluation slide to presentation
- [ ] Add risk functions to `.py` script and `.ipynb` notebook
- [ ] Ensure 4 team case studies with 3 problems each (no solutions)
- [ ] Add `references_PXX.bib` with relevant standards
- [ ] Verify build: `latexmk -xelatex -interaction=nonstopmode`

---

## 11. Python Environment

### requirements.txt (project-wide)

```
numpy
scipy
pandas
matplotlib
seaborn
scikit-image
opencv-python
notebook
jupyter
openpiv
ipywidgets
tqdm
pyarrow
fastparquet
tabulate
ipympl
plotly
kaleido
```

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Git Conventions

- **Default branch:** `main`
- **Working branch:** `Refactor-&-Style`
- **Commit messages:** English, imperative tense (e.g., "Add risk matrix to P01 manual")
- **Do not commit:** `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.xdv`, `.bbl`, `.bcf`,
  `.blg`, `.run.xml`, `.out`, `.toc`, `.nav`, `.snm`, `.vrb`, `__pycache__/`, `.venv/`
