param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
$previousTexInputs = $env:TEXINPUTS
try {
    # A trailing separator preserves the TeX installation's default search path.
    $env:TEXINPUTS = ".;./templates//;" + $previousTexInputs
    New-Item -ItemType Directory -Force build | Out-Null
    & $Python make_figures.py
    if ($LASTEXITCODE -ne 0) { throw "Figure generation failed." }
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }
    # BibTeX is run from paper/ so references.bib resolves without copying sources.
    & bibtex build/manuscript
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }
    1..2 | ForEach-Object {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass failed." }
    }
    Copy-Item -LiteralPath build/manuscript.pdf -Destination manuscript.pdf -Force
    Write-Host "Completed: $PSScriptRoot/manuscript.pdf"
} finally {
    $env:TEXINPUTS = $previousTexInputs
    Pop-Location
}
