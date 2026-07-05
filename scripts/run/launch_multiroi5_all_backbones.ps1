param(
    [string]$PythonExe = "E:\resarch\Anaconda3\envs\face_heart\python.exe",
    [string]$ProjectRoot = "E:\projects\face2",
    [string]$LauncherLog = ""
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable does not exist: $PythonExe"
}

$outputRoot = Join-Path $ProjectRoot "experiments\ROI_Fusion_500"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($LauncherLog)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LauncherLog = Join-Path $outputRoot "multiroi5_all_backbones_scheduler_$stamp.log"
}

$configs = @(
    "config\train\roi_fusion\nyha_3class_multiroi5_shared_resnet18_concat_weightedce.yaml",
    "config\train\roi_fusion\nyha_3class_multiroi5_shared_resnet34_concat_weightedce.yaml",
    "config\train\roi_fusion\nyha_3class_multiroi5_shared_resnet50_concat_weightedce.yaml"
)

$runScript = Join-Path $ProjectRoot "scripts\run\run_exp_roi_fusion_nyha3class_5fold.py"

foreach ($relativeConfig in $configs) {
    $config = Join-Path $ProjectRoot $relativeConfig
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] START $config" | Tee-Object -FilePath $LauncherLog -Append

    & $PythonExe $runScript --config $config 2>&1 | Tee-Object -FilePath $LauncherLog -Append

    if ($LASTEXITCODE -ne 0) {
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$stamp] FAILED exit_code=$LASTEXITCODE config=$config" | Tee-Object -FilePath $LauncherLog -Append
        exit $LASTEXITCODE
    }

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] DONE $config" | Tee-Object -FilePath $LauncherLog -Append
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$stamp] ALL_DONE" | Tee-Object -FilePath $LauncherLog -Append
