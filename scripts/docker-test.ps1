[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

docker compose run --rm test
exit $LASTEXITCODE
