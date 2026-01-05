# 停止并删除启动的后台作业
# 使用方法：在 PowerShell 中运行：
#   ./scripts/stop_services.ps1

$jobs = @("discriptor-app", "ht-app")
foreach ($j in $jobs) {
    $job = Get-Job -Name $j -ErrorAction SilentlyContinue
    if ($job) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -ErrorAction SilentlyContinue
        Write-Host "Stopped and removed job: $j"
    } else {
        Write-Host "Job not found: $j"
    }
}
