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
    # Prefer this build's auxiliary files over root-level files from an editor.
    $env:TEXINPUTS = "./build/;.;" + $savedTexInputs
    New-Item -ItemType Directory -Force -Path build | Out-Null
    & $pdfLaTeX -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }
    & $bibTeX build/manuscript
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }
    foreach ($pass in 1..2) {
        & $pdfLaTeX -interaction=nonstopmode -halt-on-error -no-shell-escape -output-directory=build manuscript.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $pass failed." }
    }
    # Replace the output file instead of overwriting its contents: PDF previews
    # may memory-map the current file and prevent an in-place write on Windows.
    $publishedPdf = Join-Path $PSScriptRoot "manuscript.pdf"
    $completedPdf = Join-Path $PSScriptRoot "build/manuscript.pdf"
    $stagedPdf = Join-Path $PSScriptRoot (".manuscript." + [Guid]::NewGuid().ToString("N") + ".pdf")
    try {
        Copy-Item -LiteralPath build/manuscript.pdf -Destination $stagedPdf
        if (Test-Path -LiteralPath $publishedPdf) {
            [System.IO.File]::Replace($stagedPdf, $publishedPdf, [NullString]::Value)
        } else {
            [System.IO.File]::Move($stagedPdf, $publishedPdf)
        }
        $completedPdf = $publishedPdf
    } catch {
        $publishError = $_.Exception
        while ($publishError.InnerException) {
            $publishError = $publishError.InnerException
        }
        $fileErrorCode = $publishError.HResult -band 0xFFFF
        if ($publishError -isnot [System.IO.IOException] -or $fileErrorCode -notin @(32, 33, 1224)) {
            throw
        }
        Write-Warning "The PDF compiled successfully, but manuscript.pdf is in use. Latest PDF: $completedPdf"
    } finally {
        if (Test-Path -LiteralPath $stagedPdf) {
            Remove-Item -LiteralPath $stagedPdf
        }
    }
    Write-Host "Completed: $completedPdf"
} finally {
    $env:TEXINPUTS = $savedTexInputs
    Pop-Location
}
