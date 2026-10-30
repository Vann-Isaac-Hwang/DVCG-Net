# run_cpsc2018_preprocess.ps1
# Run from any directory with:
# powershell -ExecutionPolicy Bypass -File .\scripts\runners\windows\run_cpsc2018_preprocess.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$PreprocessScript = Join-Path $ProjectDir "scripts/preprocessing/preprocess_cpsc2018_sr100_freq10.py"
Set-Location $ProjectDir

$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

# Your current CPSC2018 raw data directory. It should contain A0001.mat, A0002.mat, ...
$RawDir = if ($env:CPSC_RAW_DIR) { $env:CPSC_RAW_DIR } else { Join-Path $ProjectDir "data/CPSC_Data/raw" }

# Auto-detect REFERENCE file. If auto-detection fails, manually set this to the exact path.
$ReferenceCsv = ""
$ReferenceCandidates = @(
    "REFERENCE.csv",
    "REFERENCE.CSV",
    "reference.csv",
    "REFERENCE-v3.csv",
    "REFERENCE-v3.CSV",
    "REFERENCE.txt",
    "reference.txt"
)
foreach ($name in $ReferenceCandidates) {
    $p = Join-Path $RawDir $name
    if (Test-Path $p) {
        $ReferenceCsv = $p
        break
    }
}
if ([string]::IsNullOrWhiteSpace($ReferenceCsv)) {
    $found = Get-ChildItem $RawDir -File -Recurse |
        Where-Object { $_.Name -match "REFERENCE|reference|\.csv$|\.txt$" } |
        Select-Object -First 1
    if ($null -ne $found) { $ReferenceCsv = $found.FullName }
}

# Output processed data compatible with --dataset CPSC.
$OutDir = if ($env:CPSC_DATA_DIR) { $env:CPSC_DATA_DIR } else { Join-Path $ProjectDir "data/CPSC_Data/processed_sr100_freq10" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    chcp 65001 | Out-Null
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python command not found: $Python" }
if (-not (Test-Path $RawDir)) { throw "RawDir not found: $RawDir" }
if ([string]::IsNullOrWhiteSpace($ReferenceCsv) -or -not (Test-Path $ReferenceCsv)) {
    throw "REFERENCE file not found under $RawDir. Please put REFERENCE.csv there, or set `$ReferenceCsv manually in this script."
}
if (-not (Test-Path $PreprocessScript)) {
    throw "CPSC preprocessing script not found: $PreprocessScript"
}

Write-Host "ProjectDir   = $ProjectDir"
Write-Host "Python       = $Python"
Write-Host "RawDir       = $RawDir"
Write-Host "ReferenceCsv = $ReferenceCsv"
Write-Host "OutDir       = $OutDir"

& $Python -X utf8 $PreprocessScript `
  --raw_dir $RawDir `
  --reference_csv $ReferenceCsv `
  --out_dir $OutDir `
  --raw_fs 500 `
  --target_fs 100 `
  --seconds 10 `
  --cutoff 10

if ($LASTEXITCODE -ne 0) { throw "CPSC2018 preprocessing failed." }

Write-Host ""
Write-Host "CPSC2018 preprocessing finished. Checking output tensors..."
& $Python -X utf8 -c "import torch, os; p=r'$OutDir'; names=['x_lf.pth','x_hf.pth','labels.pth','ages.pth','sexes.pth']; print('OutDir=',p); [print(n, tuple(torch.load(os.path.join(p,n), weights_only=True).shape), torch.load(os.path.join(p,n), weights_only=True).dtype) for n in names]"

Write-Host ""
Write-Host "OK. Use this DataDir for CPSC experiments: $OutDir"
