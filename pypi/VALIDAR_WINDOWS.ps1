param([string]$Python = 'python', [string]$Instancias = (Join-Path $PSScriptRoot '..\TaillardInstancesJSON'))
$ErrorActionPreference = 'Stop'
$raiz = $PSScriptRoot
$ejecucion = Join-Path $raiz ('validacion-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $ejecucion | Out-Null
$env:TEMP = $ejecucion
$env:TMP = $ejecucion
Start-Transcript -Path (Join-Path $ejecucion 'validacion.log')
function Ejecutar {
    param([string]$Programa, [string[]]$Argumentos)
    & $Programa @Argumentos
    if ($LASTEXITCODE -ne 0) { throw "Error $LASTEXITCODE al ejecutar $Programa $Argumentos" }
}
try {
    $entorno = Join-Path $ejecucion 'entorno'
    Ejecutar $Python @('-m', 'venv', $entorno)
    $interprete = Join-Path $entorno 'Scripts\python.exe'
    Ejecutar $interprete @('-m', 'pip', 'install', '--upgrade', 'pip')
    Ejecutar $interprete @('-m', 'pip', 'install', "$raiz[graphs,dev,research]")
    Ejecutar $interprete @('-m', 'pip', 'check')
    Ejecutar $interprete @('-I', (Join-Path $raiz 'verify_release.py'))
    Ejecutar $interprete @('-I', (Join-Path $raiz 'validate_instance_formats.py'), (Join-Path $raiz '..'), '--output', (Join-Path $ejecucion 'instance-formats.json'))
    Ejecutar $interprete @('-I', (Join-Path $raiz 'validate_real_inputs.py'), $Instancias, '--output', (Join-Path $ejecucion 'real-inputs.json'))
    Ejecutar $interprete @('-I', (Join-Path $raiz 'validate_pipeline.py'), $Instancias, '--output', (Join-Path $ejecucion 'pipeline.json'))
    $dist = Join-Path $ejecucion 'dist'
    Ejecutar $interprete @('-m', 'build', $raiz, '--outdir', $dist)
    $archivos = @(Get-ChildItem -LiteralPath $dist -File | ForEach-Object FullName)
    Ejecutar $interprete (@('-m', 'twine', 'check', '--strict') + $archivos)
    foreach ($tipo in @('wheel', 'sdist')) {
        $smoke = Join-Path $ejecucion $tipo
        Ejecutar $Python @('-m', 'venv', $smoke)
        $smokePython = Join-Path $smoke 'Scripts\python.exe'
        if ($tipo -eq 'wheel') {
            $archivo = (Get-ChildItem -LiteralPath $dist -Filter '*.whl' | Select-Object -First 1).FullName
        } else {
            $archivo = (Get-ChildItem -LiteralPath $dist -Filter '*.tar.gz' | Select-Object -First 1).FullName
        }
        Ejecutar $smokePython @('-m', 'pip', 'install', "$archivo[graphs,research]")
        Ejecutar $smokePython @('-m', 'pip', 'check')
        Ejecutar $smokePython @('-I', (Join-Path $raiz 'verify_release.py'))
        Ejecutar $smokePython @('-I', (Join-Path $raiz 'validate_pipeline.py'), $Instancias, '--output', (Join-Path $ejecucion "$tipo-pipeline.json"))
    }
    Ejecutar $interprete @('-m', 'pip', 'freeze')
    Write-Output "VALIDACION COMPLETA SIN OMISIONES. Resultados: $ejecucion"
} finally {
    Stop-Transcript
}
