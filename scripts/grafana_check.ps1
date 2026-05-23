$creds = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:admin"))
$headers = @{ Authorization = "Basic $creds" }

Write-Host "=== Datasources ==="
$ds = Invoke-WebRequest -Uri "http://localhost:3000/api/datasources" -Headers $headers -UseBasicParsing
Write-Host $ds.Content

Write-Host "=== Dashboards ==="
$db = Invoke-WebRequest -Uri "http://localhost:3000/api/search?type=dash-db" -Headers $headers -UseBasicParsing
Write-Host $db.Content
