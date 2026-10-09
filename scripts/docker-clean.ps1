[CmdletBinding()]
param(
    [switch]$Volumes,
    [switch]$BuildCache
)

$ErrorActionPreference = "Stop"

$containers = docker ps -aq
if ($containers) {
    docker rm -f $containers
}

docker image prune -a -f
docker network prune -f
if ($Volumes) {
    docker volume prune -f
}
if ($BuildCache) {
    docker builder prune -a -f
}

Write-Host "Docker cleanup completed."
