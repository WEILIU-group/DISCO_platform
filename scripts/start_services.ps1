# 启动两个 Dash 服务（以后台 Job 形式），并将日志写入 ../logs
# 使用方法：在 PowerShell 中运行：
#   ./scripts/start_services.ps1

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Resolve-Path "$base\.."
$logs = Join-Path $project "logs"
if (-Not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }

# 启动 SISSO (discriptor/app.py) -> 8050
Start-Job -Name "discriptor-app" -ScriptBlock {
    cd "C:\Users\hp\Desktop\platform"
    python "C:\Users\hp\Desktop\platform\discriptor\app.py" *>&1 | Out-File -FilePath "C:\Users\hp\Desktop\platform\logs\discriptor.log" -Encoding utf8 -Append
}

# 启动 High-throughput (high-throught-calcultion/app.py) -> 8051
Start-Job -Name "ht-app" -ScriptBlock {
    cd "C:\Users\hp\Desktop\platform"
    python "C:\Users\hp\Desktop\platform\high-throught-calcultion\app.py" *>&1 | Out-File -FilePath "C:\Users\hp\Desktop\platform\logs\ht-app.log" -Encoding utf8 -Append
}

Write-Host "Started jobs: discriptor-app, ht-app. Logs: ./logs/discriptor.log, ./logs/ht-app.log" -ForegroundColor Green
