[CmdletBinding()]
param(
    [string]$Image = "sp35-bridge-inspection",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

Write-Host "Building runtime image $Image`:$Tag"
docker build --pull --target runtime -t "$Image`:$Tag" .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building test image $Image`:test"
docker build --pull --target test -t "$Image`:test" .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Images built successfully."
