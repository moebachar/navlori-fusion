# NavLoRI Fusion - Service management for InfluxDB + Grafana
# Usage: .\scripts\services.ps1 [start|stop|status]

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "status"
)

$FusionDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ServicesDir = Join-Path $FusionDir "services"
$InfluxdbExe = Join-Path $ServicesDir "influxdb\influxd.exe"
$GrafanaExe = Join-Path $ServicesDir "grafana\bin\grafana-server.exe"
$GrafanaHome = Join-Path $ServicesDir "grafana"

function Start-InfluxDB {
    $existing = Get-Process -Name "influxd" -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "[InfluxDB] Already running (PID $($existing.Id))"
        return
    }
    Write-Host "[InfluxDB] Starting on http://localhost:8086 ..."
    $boltPath = Join-Path $ServicesDir "influxdb\influxd.bolt"
    $enginePath = Join-Path $ServicesDir "influxdb\engine"
    $logPath = Join-Path $ServicesDir "influxdb.log"
    $proc = Start-Process -FilePath $InfluxdbExe `
        -ArgumentList "--bolt-path=`"$boltPath`"", "--engine-path=`"$enginePath`"" `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError (Join-Path $ServicesDir "influxdb_err.log") `
        -WindowStyle Hidden -PassThru
    Write-Host "[InfluxDB] Started (PID $($proc.Id))"
}

function Start-Grafana {
    $existing = Get-Process -Name "grafana-server" -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "[Grafana] Already running (PID $($existing.Id))"
        return
    }
    Write-Host "[Grafana] Starting on http://localhost:3000 ..."
    $logPath = Join-Path $ServicesDir "grafana.log"
    $proc = Start-Process -FilePath $GrafanaExe `
        -ArgumentList "--homepath=`"$GrafanaHome`"" `
        -WorkingDirectory $GrafanaHome `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError (Join-Path $ServicesDir "grafana_err.log") `
        -WindowStyle Hidden -PassThru
    Write-Host "[Grafana] Started (PID $($proc.Id))"
}

function Stop-Service-ByName {
    param([string]$ProcessName, [string]$DisplayName)
    $procs = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force
        Write-Host "[$DisplayName] Stopped"
    } else {
        Write-Host "[$DisplayName] Not running"
    }
}

function Get-ServiceStatus {
    param([string]$ProcessName, [string]$DisplayName)
    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "[$DisplayName] Running (PID $($proc.Id))"
    } else {
        Write-Host "[$DisplayName] Stopped"
    }
}

switch ($Action) {
    "start" {
        Start-InfluxDB
        Start-Grafana
        Write-Host ""
        Write-Host "Services:"
        Write-Host "  InfluxDB  -> http://localhost:8086"
        Write-Host "  Grafana   -> http://localhost:3000 (admin/admin)"
    }
    "stop" {
        Stop-Service-ByName "influxd" "InfluxDB"
        Stop-Service-ByName "grafana-server" "Grafana"
    }
    "status" {
        Get-ServiceStatus "influxd" "InfluxDB"
        Get-ServiceStatus "grafana-server" "Grafana"
    }
}
