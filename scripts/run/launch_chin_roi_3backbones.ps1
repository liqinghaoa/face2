param(
    [string]$ProjectRoot = "E:\projects\face2",
    [string]$PythonExe = "E:\resarch\Anaconda3\envs\face_heart\python.exe",
    [string]$CudaVisibleDevices = "0"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices

$RunScript = Join-Path $ProjectRoot "scripts\run\run_exp_roi_nyha3class_5fold.py"
$LogDir = Join-Path $ProjectRoot "experiments\ROI_500\_launcher_logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "chin_roi_resnet18_34_50_$Stamp.log"
$Backbones = @("resnet18", "resnet34", "resnet50")

"START chin_roi 3-backbone training: $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
"ProjectRoot=$ProjectRoot" | Tee-Object -FilePath $LogPath -Append
"PythonExe=$PythonExe" | Tee-Object -FilePath $LogPath -Append
"CUDA_VISIBLE_DEVICES=$CudaVisibleDevices" | Tee-Object -FilePath $LogPath -Append

foreach ($Backbone in $Backbones) {
    "===== START chin_roi ${Backbone}: $(Get-Date -Format o) =====" | Tee-Object -FilePath $LogPath -Append
    & $PythonExe $RunScript --roi chin_roi --backbone $Backbone 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    "===== END chin_roi ${Backbone} exit=${ExitCode}: $(Get-Date -Format o) =====" | Tee-Object -FilePath $LogPath -Append

    if ($ExitCode -ne 0) {
        "ABORT because ${Backbone} failed." | Tee-Object -FilePath $LogPath -Append
        exit $ExitCode
    }
}

"FINISH chin_roi 3-backbone training: $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
