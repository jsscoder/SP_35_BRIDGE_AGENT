[CmdletBinding()]
param(
    [string]$Username = "jhoijlk",
    [string]$Repository = "sp35-bridge-inspection",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$image = "$Username/$Repository`:$Tag"

docker build --pull --target runtime -t $image .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker push $image
exit $LASTEXITCODE
