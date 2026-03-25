# Launch Webots in the interactive desktop session (from SSH terminal)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/launch_webots.ps1 [start|stop|status]

param([string]$Action = "start")

$WebotsExe = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
$WorldFile = "C:\Users\Administrateur\navlori-fusion\src\simulation\worlds\Tiago++'s world.wbt"
$TaskName = "NavLoRI_Webots"

switch ($Action) {
    "start" {
        # Kill any existing Webots
        Get-Process webots -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep 1

        # Remove old task if exists
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

        # Create task that runs in interactive desktop session
        $act = New-ScheduledTaskAction -Execute $WebotsExe -Argument "--batch --minimize --mode=fast `"$WorldFile`""
        $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        $principal = New-ScheduledTaskPrincipal -UserId "Administrateur" -LogonType Interactive -RunLevel Highest
        Register-ScheduledTask -TaskName $TaskName -Action $act -Settings $set -Principal $principal -Force | Out-Null
        Start-ScheduledTask -TaskName $TaskName

        Start-Sleep 3
        $proc = Get-Process webots -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[Webots] Running (PID $($proc.Id))" -ForegroundColor Green
            Write-Host "[Webots] World: $WorldFile"
            Write-Host "[Webots] Mode: batch, minimize, fast (rendering enabled for cameras)"
            Write-Host "[Webots] To stop: powershell -ExecutionPolicy Bypass -File scripts/launch_webots.ps1 stop"
        } else {
            Write-Host "[Webots] Failed to start" -ForegroundColor Red
        }
    }
    "stop" {
        Get-Process webots -ErrorAction SilentlyContinue | Stop-Process -Force
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "[Webots] Stopped" -ForegroundColor Yellow
    }
    "status" {
        $proc = Get-Process webots -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[Webots] Running (PID $($proc.Id))" -ForegroundColor Green
        } else {
            Write-Host "[Webots] Not running" -ForegroundColor Yellow
        }
    }
}
