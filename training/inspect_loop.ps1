# Periodic real-play inspection during training (Windows PowerShell).
#   powershell -File training\inspect_loop.ps1 -Run aim1 -EveryMin 45
# Every N minutes: snapshot the run's latest checkpoint and play all held-out maps with
# training.inspect_play (videos, top-down paths, frame sheets, behaviour stats) into
# videos\inspect_<run>_<HHmm>\, then print an INSPECT_DONE line to logs\inspect_<run>.log.
param([string]$Run = "aim1", [int]$EveryMin = 45)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = "logs\inspect_$Run.log"
while ($true) {
    Start-Sleep -Seconds ($EveryMin * 60)
    $tag = "$Run" + "_" + (Get-Date -Format "HHmm")
    $snap = "checkpoints\$Run\inspect_$tag.pt"
    Copy-Item "checkpoints\$Run\latest.pt" $snap
    "INSPECT_START $tag $(Get-Date -Format HH:mm)" | Out-File -Append -Encoding utf8 $log
    & .\.venv\Scripts\python.exe -m training.inspect_play --ckpt $snap --tag $tag --gif-maps "heldout_s1_1,heldout_s3_5,heldout_s4_6" 2>&1 |
        Select-Object -Last 12 | Out-File -Append -Encoding utf8 $log
    "INSPECT_DONE $tag videos\inspect_$tag" | Out-File -Append -Encoding utf8 $log
}
