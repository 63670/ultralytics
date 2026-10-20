param(
    [string]$TeXBin = ""
)

$ErrorActionPreference = "Stop"
$savedTexInputs = $env:TEXINPUTS
Push-Location $PSScriptRoot
try {
    # Resolve tools without regenerating or modifying manuscript figures.
    $pdfLaTeX = "pdflatex"
    $bibTeX = "bibtex"
    if ($TeXBin) {
        $pdfLaTeX = Join-Path $TeXBin "pdflatex.exe"
        $bibTeX = Join-Path $TeXBin "bibtex.exe"
    }
    $env:TEXINPUTS = ".;" + $savedTexInputs
    New-Item -ItemType Directory -Force -Path build | Out-Null
    & $pdfLaTeX -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }
    & $bibTeX build/manuscript
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }
    foreach ($pass in 1..2) {
        & $pdfLaTeX -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $pass failed." }
    }
    Copy-Item -LiteralPath build/manuscript.pdf -Destination manuscript.pdf -Force
    Write-Host "Completed: $PSScriptRoot/manuscript.pdf"
} finally {
    $env:TEXINPUTS = $savedTexInputs
    Pop-Location
}
