$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
  $python = ".\.venv\Scripts\python.exe"
}

& $python ".\run_once.py" "--config" ".\config.json"
