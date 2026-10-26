param([string]$Python = 'python', [int]$MaxSize = 15, [int]$Trees = 100)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path $PSScriptRoot ('resultados-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $out | Out-Null
$env:TEMP = $out
$env:TMP = $out
Start-Transcript -Path (Join-Path $out 'execution.log')
function Run {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Error $LASTEXITCODE al ejecutar $Python $Arguments" }
}
try {
    Run @('-m', 'pip', 'install', "$root\pypi[graphs,dev]")
    Run @('-m', 'pip', 'check')
    Run @('-I', (Join-Path $root 'pypi\verify_release.py'))
    Run @('-m', 'unittest', 'discover', '-s', $PSScriptRoot, '-p', 'test_study.py', '-v')
    Run @('-I', (Join-Path $PSScriptRoot 'run_study.py'), '--root', $root, '--out', $out, '--baselines-only')
    Run @('-I', (Join-Path $PSScriptRoot 'run_study.py'), '--root', $root, '--out', $out, '--max-size', "$MaxSize", '--trees', "$Trees")
    Run @('-m', 'pip', 'freeze')
    Write-Output "ESTUDIO CEJSP COMPLETO. Resultados: $out"
} finally {
    Stop-Transcript
}
