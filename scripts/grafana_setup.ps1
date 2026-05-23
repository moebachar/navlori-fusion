# Add InfluxDB datasource and import NavLoRI dashboards into Grafana via API.
#
# Reads credentials from environment variables (load .env first if needed):
#   INFLUXDB_TOKEN    — required
#   GRAFANA_USER      — default "admin"
#   GRAFANA_PASSWORD  — default "admin"
#   GRAFANA_URL       — default "http://localhost:3000"

if (-not $env:INFLUXDB_TOKEN) {
    Write-Host "[Grafana] INFLUXDB_TOKEN must be set in the environment."
    Write-Host "          See .env.example at the repo root."
    exit 1
}

$base = if ($env:GRAFANA_URL) { $env:GRAFANA_URL } else { "http://localhost:3000" }
$user = if ($env:GRAFANA_USER) { $env:GRAFANA_USER } else { "admin" }
$pass = if ($env:GRAFANA_PASSWORD) { $env:GRAFANA_PASSWORD } else { "admin" }

$creds = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${user}:${pass}"))
$headers = @{ Authorization = "Basic $creds"; "Content-Type" = "application/json" }

# 1) Add InfluxDB datasource
Write-Host "[1/2] Adding InfluxDB datasource ..."
$ds = @{
    name      = "InfluxDB"
    type      = "influxdb"
    access    = "proxy"
    url       = "http://localhost:8086"
    isDefault = $true
    jsonData  = @{
        version       = "Flux"
        organization  = "navlori"
        defaultBucket = "async_data"
    }
    secureJsonData = @{
        token = $env:INFLUXDB_TOKEN
    }
}
$dsBody = $ds | ConvertTo-Json -Depth 5
$r = Invoke-WebRequest -Uri "$base/api/datasources" -Method POST -Headers $headers -Body $dsBody -UseBasicParsing
Write-Host "[OK] Datasource: $($r.StatusCode)"

# 2) Import each dashboard JSON. Resolve dashboard dir relative to this script
# so the path stays portable across machines.
$repoRoot = Split-Path -Parent $PSScriptRoot
$dashDir  = Join-Path $repoRoot "src\services\grafana\dashboards"
$jsons = Get-ChildItem -Path $dashDir -Filter "*.json"
foreach ($f in $jsons) {
    Write-Host "[2/?] Importing $($f.Name) ..."
    $raw = Get-Content $f.FullName -Raw
    $importBody = "{`"dashboard`": $raw, `"overwrite`": true, `"folderId`": 0}"
    $r2 = Invoke-WebRequest -Uri "$base/api/dashboards/import" -Method POST -Headers $headers -Body $importBody -UseBasicParsing
    Write-Host "[OK] $($f.Name): $($r2.StatusCode)"
}

Write-Host ""
Write-Host "Done. Open $base/dashboards to see them."
