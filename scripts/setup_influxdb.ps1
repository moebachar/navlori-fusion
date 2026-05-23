# Initialize InfluxDB with the org, bucket, and token the controllers expect.
#
# Reads credentials from environment variables (load .env first if needed).
# Required: INFLUXDB_TOKEN, INFLUXDB_PASSWORD
# Optional: INFLUXDB_USERNAME (default "admin"), INFLUXDB_ORG (default "navlori"),
#           INFLUXDB_BUCKET (default "async_data")
#
# Example:
#   $env:INFLUXDB_TOKEN  = "<paste-token-here>"
#   $env:INFLUXDB_PASSWORD = "<paste-password-here>"
#   pwsh -File scripts\setup_influxdb.ps1

if (-not $env:INFLUXDB_TOKEN -or -not $env:INFLUXDB_PASSWORD) {
    Write-Host "[InfluxDB] INFLUXDB_TOKEN and INFLUXDB_PASSWORD must be set in the environment."
    Write-Host "           See .env.example at the repo root."
    exit 1
}

$username = if ($env:INFLUXDB_USERNAME) { $env:INFLUXDB_USERNAME } else { "admin" }
$org      = if ($env:INFLUXDB_ORG)      { $env:INFLUXDB_ORG }      else { "navlori" }
$bucket   = if ($env:INFLUXDB_BUCKET)   { $env:INFLUXDB_BUCKET }   else { "async_data" }

$payload = @{
    username              = $username
    password              = $env:INFLUXDB_PASSWORD
    org                   = $org
    bucket                = $bucket
    token                 = $env:INFLUXDB_TOKEN
    retentionPeriodSeconds = 0
} | ConvertTo-Json -Compress

try {
    $r = Invoke-WebRequest -Uri "http://localhost:8086/api/v2/setup" -Method POST -Body $payload -ContentType "application/json" -UseBasicParsing
    Write-Host "[InfluxDB] Setup OK (status $($r.StatusCode))"
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 422) {
        Write-Host "[InfluxDB] Already configured, nothing to do."
    } else {
        Write-Host "[InfluxDB] Setup failed, check if influxd is running."
    }
}
