# LaTeX Build Requirements – Manuals & Presentations

## Compiler Toolchain

| Tool       | Minimum version | Purpose                        |
|------------|-----------------|--------------------------------|
| **XeLaTeX** | ≥ 4.x (MiKTeX 25+) | Main TeX engine (UTF-8 native, `fontspec` support) |
| **Biber**   | ≥ 2.17         | Bibliography backend for `biblatex` |
| **Latexmk** | ≥ 4.80         | Build automation (dependency tracking, multi-pass) |

> **Important:** These documents require **XeLaTeX**, not pdfLaTeX.  
> The preambles use `\usepackage{fontspec}` which is XeLaTeX/LuaLaTeX-only.

---

## One-liner build commands

### Single manual (e.g. P01)

```powershell
cd manuals\P01_Proteccion_Laser
latexmk -xelatex -interaction=nonstopmode manual_P01.tex
```

### Single presentation (e.g. P01)

```powershell
cd manuals\P01_Proteccion_Laser\presentation
latexmk -xelatex -interaction=nonstopmode presentacion_P01.tex
```

### Build all manuals and presentations (P01–P04)

```powershell
$practices = @("P01_Proteccion_Laser","P02_Calibracion_Tunel","P03_Hilo_Caliente","P04_LDA_Perfil_Velocidad")
$root = "manuals"

foreach ($p in $practices) {
    $num = $p.Substring(1,2)   # "01", "02", …
    # Manual
    Push-Location "$root\$p"
    latexmk -xelatex -interaction=nonstopmode "manual_P$num.tex"
    Pop-Location
    # Presentation
    Push-Location "$root\$p\presentation"
    latexmk -xelatex -interaction=nonstopmode "presentacion_P$num.tex"
    Pop-Location
}
```

### Clean auxiliary files

```powershell
latexmk -C          # in the directory containing the .tex file
```

---

## Required LaTeX Packages

All packages below are available in standard MiKTeX / TeX Live distributions.
MiKTeX will auto-install missing packages on first compile if the option is enabled.

### Core (all documents)

| Package | Role |
|---------|------|
| `fontspec` | OpenType/TrueType font selection (XeLaTeX) |
| `babel` (spanish) | Spanish hyphenation and active characters |
| `amsmath`, `amssymb`, `amsfonts` | Mathematical typesetting |
| `siunitx` | SI units formatting |
| `tikz` | Vector graphics |
| `tikz` library **`babel`** | Fixes TikZ `->` arrow syntax conflict with Spanish babel |
| `booktabs` | Professional table rules |
| `listings` | Source code formatting |
| `hyperref` | Clickable cross-references and URLs |
| `xcolor` | Color support |
| `graphicx` | Image inclusion |

### Manual-specific

| Package | Role |
|---------|------|
| `mathtools` | Extended math (e.g. `\Aboxed`) |
| `geometry` | Page margins |
| `fancyhdr` | Custom headers/footers |
| `biblatex` + `biber` | IEEE-style bibliography |
| `float` | `[H]` float placement |
| `caption`, `subcaption` | Figure captions |
| `multirow`, `array` | Advanced table layouts |

### Presentation-specific

| Package | Role |
|---------|------|
| `beamer` (Madrid theme, seahorse color) | Slide framework |
| `colortbl` | `\rowcolor` support in tables |

---

## Key preamble patterns

### XeLaTeX encoding (replaces `inputenc`/`fontenc`)

```latex
\usepackage{fontspec}          % instead of \usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}    %            \usepackage[T1]{fontenc}
```

### TikZ + Spanish babel fix

```latex
\usepackage{tikz}
\usetikzlibrary{babel}         % ← MUST come before other tikz libraries
\usetikzlibrary{arrows.meta,positioning,calc,...}
```

Without `\usetikzlibrary{babel}`, Spanish babel's active `>` and `<` characters
break TikZ arrow syntax (`->`, `<->`, etc.).

### S-column literals in `siunitx` tables

Non-numeric content in `S` columns must be wrapped in braces:

```latex
\begin{tabular}{lSS}
    ...
    $OD_{\min}$  & 5.71  & {--} \\   % ← braces around --
\end{tabular}
```

### Multi-line text in TikZ nodes

TikZ nodes using `\\` require the `align` key:

```latex
\node[align=center] at (x,y) {Line 1\\Line 2};
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Argument of \language@active@arg> has an extra }` | Spanish babel vs TikZ arrows | Add `\usetikzlibrary{babel}` |
| `\Aboxed undefined` | Missing `mathtools` | Add `\usepackage{mathtools}` |
| `Invalid number '--'` in S-column | `siunitx` parsing literal text | Wrap in braces: `{--}` |
| `Not allowed in LR mode` on `\\` in TikZ | Missing `align` key on node | Add `align=center` to node options |
| `\rowcolor undefined` (Beamer) | Missing `colortbl` | Add `\usepackage{colortbl}` |
| `fontspec` errors with pdfLaTeX | Wrong engine | Use `xelatex` or `latexmk -xelatex` |

---

## File structure

```
manuals/
├── BUILD.md                          ← this file
├── P01_Proteccion_Laser/
│   ├── manual_P01.tex
│   ├── references_P01.bib
│   └── presentation/
│       └── presentacion_P01.tex
├── P02_Calibracion_Tunel/
│   ├── manual_P02.tex
│   ├── references_P02.bib
│   └── presentation/
│       └── presentacion_P02.tex
├── P03_Hilo_Caliente/
│   ├── manual_P03.tex
│   ├── references_P03.bib
│   └── presentation/
│       └── presentacion_P03.tex
└── P04_LDA_Perfil_Velocidad/
    ├── manual_P04.tex
    ├── references_P04.bib
    └── presentation/
        └── presentacion_P04.tex
```
