# NavLoRI — One-shot InfluxDB + Grafana setup
# Run once from the Parsec desktop as Administrator
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_services.ps1

$Root      = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SvcDir    = Join-Path $Root "services"
$TmpDir    = Join-Path $Root "services\_tmp"

$InfluxUrl = "https://dl.influxdata.com/influxdb/releases/influxdb-2.7.10_windows_amd64.zip"
$GrafanaUrl= "https://dl.grafana.com/oss/release/grafana-11.1.0.windows-amd64.zip"

New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

# ── InfluxDB ──────────────────────────────────────────────────────────────
$InfluxDest = Join-Path $SvcDir "influxdb"
if (-Not (Test-Path (Join-Path $InfluxDest "influxd.exe"))) {
    Write-Host "[1/4] Downloading InfluxDB 2.7.10 ..."
    $zip = Join-Path $TmpDir "influxdb.zip"
    Invoke-WebRequest -Uri $InfluxUrl -OutFile $zip -UseBasicParsing
    Write-Host "[2/4] Extracting InfluxDB ..."
    Expand-Archive -Path $zip -DestinationPath $TmpDir -Force
    $extracted = Get-ChildItem $TmpDir -Directory | Where-Object { $_.Name -like "influxdb*" } | Select-Object -First 1
    Move-Item -Path $extracted.FullName -Destination $InfluxDest -Force
    Remove-Item $zip
    Write-Host "[OK] InfluxDB ready at $InfluxDest"
} else {
    Write-Host "[OK] InfluxDB already present, skipping download."
}

# ── Grafana ───────────────────────────────────────────────────────────────
$GrafanaDest = Join-Path $SvcDir "grafana"
if (-Not (Test-Path (Join-Path $GrafanaDest "bin\grafana-server.exe"))) {
    Write-Host "[3/4] Downloading Grafana 11.1.0 ..."
    $zip = Join-Path $TmpDir "grafana.zip"
    Invoke-WebRequest -Uri $GrafanaUrl -OutFile $zip -UseBasicParsing
    Write-Host "[4/4] Extracting Grafana ..."
    Expand-Archive -Path $zip -DestinationPath $TmpDir -Force
    $extracted = Get-ChildItem $TmpDir -Directory | Where-Object { $_.Name -like "grafana*" } | Select-Object -First 1
    Move-Item -Path $extracted.FullName -Destination $GrafanaDest -Force
    Remove-Item $zip
    Write-Host "[OK] Grafana ready at $GrafanaDest"
} else {
    Write-Host "[OK] Grafana already present, skipping download."
}

# ── Copy provisioning configs ─────────────────────────────────────────────
$ProvSrc  = Join-Path $Root "src\services\grafana\provisioning"
$ProvDest = Join-Path $GrafanaDest "conf\provisioning"
$DashSrc  = Join-Path $Root "src\services\grafana\dashboards"
$DashDest = Join-Path $GrafanaDest "dashboards\navlori"

Write-Host "[+] Copying Grafana provisioning config ..."
Copy-Item -Recurse -Force $ProvSrc\* $ProvDest\
New-Item -ItemType Directory -Force -Path $DashDest | Out-Null
Copy-Item -Force $DashSrc\* $DashDest\
Write-Host "[OK] Provisioning configs applied."

# ── Cleanup ───────────────────────────────────────────────────────────────
Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Setup complete. Now run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\services.ps1 start"
