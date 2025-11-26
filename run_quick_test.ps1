# =============================================================================
# Quick Test Script (English Version)
# =============================================================================
$ErrorActionPreference = "Stop"
$Py = "python"
$Root = (Get-Location).Path
# Path to your data
$DataRoot = Join-Path $Root "preprocess\_clean\bins"
$RunsDir = Join-Path $Root "runs_quick_test"

# Run first two stages with reduced steps
$Stages = @(
    # Stage 1: 0-5 (Validate basic learning)
    @{ Name = "0-5"; Folder = "0-5"; Steps = 10000; NSteps = 512; Batch = 64; Ent = 0.02 },
  
    # Stage 2: 6-9 (Validate generalization)
    @{ Name = "6-9"; Folder = "6-9"; Steps = 10000; NSteps = 512; Batch = 64; Ent = 0.02 }
)

# --- Functions ---
function Find-Model($outDir) {
    $p = Join-Path $outDir "multi_cell_model.zip"
    if (Test-Path $p) { return $p }
    return $null
}

function Train-Stage($st, $resumeFrom) {
    # Try to find data folder (either root or train subdir)
    $tryTrain = Join-Path $DataRoot "$($st.Folder)\train"
    $tryRoot = Join-Path $DataRoot "$($st.Folder)"
    
    if (Test-Path $tryTrain) { $trainDir = $tryTrain }
    elseif (Test-Path $tryRoot) { $trainDir = $tryRoot }
    else { throw "Data folder not found for: $($st.Folder)" }

    $outDir = Join-Path $RunsDir $st.Name
    
    Write-Host "`n=== [QUICK TEST] Stage: $($st.Name) ===" -ForegroundColor Cyan
    
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $argsList = @(
        "transistor_placement.py",
        "--env-dir", $trainDir,
        "--output-dir", $outDir,
        "--timesteps", "$($st.Steps)",
        "--n-steps", "$($st.NSteps)",
        "--batch-size", "$($st.Batch)",
        "--learning-rate", "1e-4",
        "--ent-coef", "$($st.Ent)",
        "--w-share", "5.0", "--w-break", "2.0", "--w-hpwl", "0.5"
    )

    if ($resumeFrom) {
        Write-Host " -> Resuming..." -ForegroundColor Gray
        $argsList += "--resume-from"
        $argsList += $resumeFrom
    }
    else {
        Write-Host " -> Fresh Start" -ForegroundColor Gray
    }

    $p = Start-Process -FilePath $Py -ArgumentList $argsList -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Training Failed!" }

    return Find-Model $outDir
}

# --- Main ---
if (Test-Path $RunsDir) { Remove-Item $RunsDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null

$prevModel = $null

foreach ($st in $Stages) {
    try {
        $prevModel = Train-Stage $st $prevModel
        if (!$prevModel) { Write-Warning "No model found. Stopping."; break }
    }
    catch {
        Write-Error $_; break
    }
}
Write-Host "`nQuick test complete! Please check Tensorboard." -ForegroundColor Green