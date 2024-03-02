# Make sure the environment is set
& $PSScriptRoot\createenv.ps1

# Start the server
Start-Process -FilePath "python" -ArgumentList "$PSScriptRoot\main.py" -NoNewWindow -RedirectStandardOutput server_logOut.txt -RedirectStandardError server_logErr.txt

# FIXME: Make a timeout on the client
Start-Sleep -Seconds 5

# Start the client
# FIXME: Get client logs
Start-Process -FilePath "python" -ArgumentList "$PSScriptRoot\transcribe-demo.py" -NoNewWindow -RedirectStandardOutput client_logOut.txt -RedirectStandardError client_logErr.txt

Write-Output "Hello world!"