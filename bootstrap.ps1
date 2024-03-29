# Make sure the environment is set
$ROOT = $PSScriptRoot

& $ROOT\createenv.ps1

# Remove any old log directories and create a new one
Remove-Item $ROOT\log2 -Recurse -ErrorAction Ignore
New-Item -Path "$ROOT" -Name "log2" -ItemType Directory

# Start the server
#Start-Process -FilePath "python" -ArgumentList "$PSScriptRoot\main.py --record" #-NoNewWindow -RedirectStandardOutput $ROOT\log2\server_logOut.txt -RedirectStandardError $ROOT\log2\server_logErr.txt

# FIXME: Make a timeout on the client
#Start-Sleep -Seconds 5

# Start the client
# FIXME: Get client logs
#Start-Process -FilePath "python" -ArgumentList "$PSScriptRoot\transcribe-demo.py" -NoNewWindow -RedirectStandardOutput $ROOT\log\client_logOut.txt -RedirectStandardError $ROOT\log\client_logErr.txt

Write-Output "Hello world!"