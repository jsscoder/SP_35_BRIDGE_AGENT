$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
$t = [Environment]::GetEnvironmentVariable('HF_TOKEN','User')
if (-not $t) { Write-Output 'BLOCKED: no User HF_TOKEN'; exit 2 }
git clone "https://omkar:$t@huggingface.co/facebook/sam3.1" data/weights/sam3.1
cd data/weights/sam3.1
git remote set-url origin https://huggingface.co/facebook/sam3.1
Write-Output 'remote scrubbed:'
git remote -v
