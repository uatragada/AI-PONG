param(
  [string]$TaskName = "AI Pong Human Training",
  [int]$EveryMinutes = 720,
  [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$runnerDir = $PSScriptRoot
$scriptPath = Join-Path $runnerDir "run_once.ps1"

if (-not (Test-Path $scriptPath)) {
  throw "Missing runner script: $scriptPath"
}

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" `
  -WorkingDirectory $runnerDir

$trigger = New-ScheduledTaskTrigger `
  -Once `
  -At ((Get-Date).AddMinutes(5)) `
  -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
  -RepetitionDuration (New-TimeSpan -Days 9999)

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Pull AI Pong human gameplay chunks, train a supervised candidate model, and promote it when gates pass." `
  -Force | Out-Null

if ($StartNow) {
  Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Installed scheduled task '$TaskName' every $EveryMinutes minutes."
