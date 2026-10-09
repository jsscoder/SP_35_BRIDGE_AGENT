# Rebuild the GPU-worker environment .venv-win from requirements\vision-win.txt (audit A7).
# Usage (PowerShell, repo root):  .\tools\setup_vision_env.ps1          # refuses to touch an existing env
#                                 .\tools\setup_vision_env.ps1 -Force   # rebuild from scratch
param([switch]$Force)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv-win'
$req = Join-Path $root 'requirements\vision-win.txt'

if (Test-Path $venv) {
    if (-not $Force) { Write-Output ".venv-win already exists; use -Force to rebuild it."; exit 1 }
    Remove-Item -Recurse -Force $venv
}
uv venv $venv --python 3.12
uv pip install --python (Join-Path $venv 'Scripts\python.exe') -r $req --index-strategy unsafe-best-match
& (Join-Path $venv 'Scripts\python.exe') (Join-Path $root 'tools\env_check.py')
