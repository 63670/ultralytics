param(
    [string]$TeXBin = ""
)

$ErrorActionPreference = "Stop"
$savedTexInputs = $env:TEXINPUTS
$buildMutex = $null
$ownsBuildMutex = $false
Push-Location $PSScriptRoot
try {
    $sources = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.tex" -File)
    if ($sources.Count -ne 1) {
        throw "Expected exactly one manuscript .tex file in $PSScriptRoot."
    }
    $source = $sources[0]
    $documentName = $source.BaseName
    $documentHasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $documentKey = [System.BitConverter]::ToString(
            $documentHasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($source.FullName.ToLowerInvariant()))
        ).Replace("-", "").Substring(0, 24)
    } finally {
        $documentHasher.Dispose()
    }
    $buildMutex = [System.Threading.Mutex]::new($false, "Local\MCGI_LaTeX_$documentKey")
    try {
        $ownsBuildMutex = $buildMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsBuildMutex = $true
    }
    if (-not $ownsBuildMutex) {
        Write-Host "A build is already running for this manuscript; duplicate build skipped."
        return
    }
    $pdfLaTeX = "pdflatex"
    $bibTeX = "bibtex"
    if ($TeXBin) {
        $pdfLaTeX = Join-Path $TeXBin "pdflatex.exe"
        $bibTeX = Join-Path $TeXBin "bibtex.exe"
    }
    $env:TEXINPUTS = "./build/;.;" + $savedTexInputs
    New-Item -ItemType Directory -Force -Path build | Out-Null
    $compileArgs = @("-synctex=1", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape", "-output-directory=build")
    & $pdfLaTeX @compileArgs $source.Name
    if ($LASTEXITCODE -ne 0) { throw "First LaTeX pass failed." }
    & $bibTeX (Join-Path "build" $documentName)
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }
    foreach ($pass in 1..2) {
        & $pdfLaTeX @compileArgs $source.Name
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $pass failed." }
    }
    $publishedPdf = Join-Path $PSScriptRoot ($documentName + ".pdf")
    $completedPdf = Join-Path $PSScriptRoot ("build/" + $documentName + ".pdf")
    $stagedPdf = Join-Path $PSScriptRoot (".manuscript." + [Guid]::NewGuid().ToString("N") + ".pdf")
    try {
        Copy-Item -LiteralPath $completedPdf -Destination $stagedPdf
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
        Write-Warning "The PDF compiled successfully, but the published PDF is in use. Latest PDF: $completedPdf"
    } finally {
        if (Test-Path -LiteralPath $stagedPdf) {
            Remove-Item -LiteralPath $stagedPdf
        }
    }
    Write-Host "Completed: $completedPdf"
} finally {
    if ($ownsBuildMutex) {
        $buildMutex.ReleaseMutex()
    }
    if ($null -ne $buildMutex) {
        $buildMutex.Dispose()
    }
    $env:TEXINPUTS = $savedTexInputs
    Pop-Location
}
