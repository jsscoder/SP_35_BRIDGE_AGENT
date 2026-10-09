[CmdletBinding()]
param(
    [ValidateSet("core", "test")]
    [string]$Service = "core"
)

$ErrorActionPreference = "Stop"

docker compose up --pull always --abort-on-container-exit --exit-code-from $Service $Service
exit $LASTEXITCODE
