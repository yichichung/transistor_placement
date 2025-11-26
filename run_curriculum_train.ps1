# =============================================================================
# Transistor Placement Curriculum Training (V2.1 Compatible)
# =============================================================================
# Features:
# - Auto-Resume: Automatically loads the best model from the previous stage.
# - Dual-Rail Optimized Params: Tuned for the new V2.1 Python logic.
# - Reward Shaping: Prioritizes Diffusion Sharing (5.0) over HPWL (0.5) initially.
# =============================================================================

$ErrorActionPreference = "Stop"

$Py = "python"
$Root = (Get-Location).Path
# Data path (ensure prepare_data.ps1 has been run)
$DataRoot = Join-Path $Root "preprocess\_clean\bins"
$RunsDir = Join-Path $Root "runs_curriculum"

# --- Curriculum Stages ---
# LR: 1e-4 (Safe for fine-tuning after BC)
# Ent: 0.01 -> 0.002 (Decrease exploration as model matures)
$Stages = @(
    # [Phase 1: Foundation] Learn basic dual-rail placement & sharing
    @{ Name = "0-5"; Folder = "0-5"; Steps = 30000; NSteps = 512; Batch = 64; Ent = 0.02 },
    @{ Name = "6-9"; Folder = "6-9"; Steps = 40000; NSteps = 1024; Batch = 64; Ent = 0.01 },
  
    # [Mix 1] Consolidate small cells
    @{ Name = "mix_0-9"; Folder = "mix_0-9"; Steps = 60000; NSteps = 1024; Batch = 128; Ent = 0.01 },

    # [Phase 2: Growth] Medium cells
    @{ Name = "10-15"; Folder = "10-15"; Steps = 60000; NSteps = 2048; Batch = 128; Ent = 0.01 },
    @{ Name = "16-20"; Folder = "16-20"; Steps = 80000; NSteps = 2048; Batch = 128; Ent = 0.01 },

    # [Mix 2] Medium Review (Crucial step)
    @{ Name = "mix_0-20"; Folder = "mix_0-20"; Steps = 150000; NSteps = 2048; Batch = 256; Ent = 0.005 },

    # [Phase 3: Advanced] Large cells (Merged 21-50 due to sparse data)
    @{ Name = "21-50"; Folder = "21-50"; Steps = 150000; NSteps = 4096; Batch = 256; Ent = 0.005 },

    # [Final Exam] Mix All
    @{ Name = "mix_all"; Folder = "mix_all"; Steps = 300000; NSteps = 4096; Batch = 512; Ent = 0.002 }
)

# --- Helper Functions ---

function Find-Model($outDir) {
    $p = Join-Path $outDir "multi_cell_model.zip"
    if (Test-Path $p) { return $p }
    return $null
}

function Train-Stage($st, $resumeFrom) {
    # Path Logic: Try 'folder/train', fallback to 'folder'
    $tryTrain = Join-Path $DataRoot "$($st.Folder)\train"
    $tryRoot = Join-Path $DataRoot "$($st.Folder)"
    
    if (Test-Path $tryTrain) { $trainDir = $tryTrain }
    elseif (Test-Path $tryRoot) { $trainDir = $tryRoot }
    else { throw "Data folder not found: $($st.Folder). Did you run prepare_data.ps1?" }

    $outDir = Join-Path $RunsDir $st.Name
    
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "   STAGE: $($st.Name)"
    Write-Host "   Source: $trainDir"
    Write-Host "   Steps: $($st.Steps) | Ent: $($st.Ent)"
    Write-Host "========================================================" -ForegroundColor Cyan
    
    if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

    $argsList = @(
        "transistor_placement.py",
        "--env-dir", $trainDir,
        "--output-dir", $outDir,
        "--timesteps", "$($st.Steps)",
        "--n-steps", "$($st.NSteps)",
        "--batch-size", "$($st.Batch)",
        "--learning-rate", "1e-4",
        "--ent-coef", "$($st.Ent)",
        
        # --- Reward Weights (Tuned for V2.1) ---
        "--w-share", "5.0",   # High bonus for correct diffusion sharing
        "--w-break", "2.0",   # Moderate penalty for breaks
        "--w-hpwl", "0.5"  # Low penalty for wire length (initially)

    )

    if ($resumeFrom) {
        Write-Host " -> Resuming from: $resumeFrom" -ForegroundColor Yellow
        $argsList += "--resume-from"
        $argsList += $resumeFrom
    }
    else {
        Write-Host " -> Fresh Start (with BC)" -ForegroundColor Green
    }

    # Execute Python
    $p = Start-Process -FilePath $Py -ArgumentList $argsList -NoNewWindow -Wait -PassThru
    
    if ($p.ExitCode -ne 0) { 
        throw "Training Failed at stage $($st.Name)! ExitCode: $($p.ExitCode)" 
    }

    return Find-Model $outDir
}

# --- Main Loop ---

if (!(Test-Path $RunsDir)) { New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null }

$prevModel = $null

foreach ($st in $Stages) {
    try {
        $prevModel = Train-Stage $st $prevModel
        
        if (!$prevModel) {
            Write-Warning "Stage $($st.Name) finished but no model saved. Stopping."
            break
        }
        Write-Host "   [OK] Model saved." -ForegroundColor Green
    }
    catch {
        Write-Error $_
        break
    }
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "   CURRICULUM TRAINING COMPLETE"
Write-Host "   Final Model: $prevModel"
Write-Host "========================================================" -ForegroundColor Cyan