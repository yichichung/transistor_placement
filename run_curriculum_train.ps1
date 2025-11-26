# =============================================================================
# Transistor Placement Curriculum Training (Corrected Distribution)
# =============================================================================
$ErrorActionPreference = "Stop"

$Py = "python"
$Root = (Get-Location).Path
$DataRoot = Join-Path $Root "preprocess\out_cells\_clean\bins"
$RunsDir = Join-Path $Root "runs_curriculum"

# 課程規劃
# 1e-4 是配合 Behavior Cloning 最安全的學習率
$Stages = @(
    # --- Phase 1: 基礎期 (0-9) ---
    @{ Name = "0-5"; Folder = "0-5"; Steps = 30000; NSteps = 512; Batch = 64; Ent = 0.02 },
    @{ Name = "6-9"; Folder = "6-9"; Steps = 40000; NSteps = 1024; Batch = 64; Ent = 0.02 },
  
    # [Review 1] 84 files
    @{ Name = "mix_0-9"; Folder = "mix_0-9"; Steps = 60000; NSteps = 1024; Batch = 128; Ent = 0.01 },

    # --- Phase 2: 成長期 (10-20) ---
    @{ Name = "10-15"; Folder = "10-15"; Steps = 60000; NSteps = 2048; Batch = 128; Ent = 0.01 },
    @{ Name = "16-20"; Folder = "16-20"; Steps = 80000; NSteps = 2048; Batch = 128; Ent = 0.01 },

    # [Review 2] 162 files (90% 的資料都在這)
    @{ Name = "mix_0-20"; Folder = "mix_0-20"; Steps = 150000; NSteps = 2048; Batch = 256; Ent = 0.005 },

    # --- Phase 3: 進階期 (21-50) ---
    # 使用合併後的資料夾 (共 18 個檔案)
    # 因為是大電路，Steps 拉長，NSteps 加大以求穩定
    @{ Name = "21-50"; Folder = "21-50"; Steps = 150000; NSteps = 4096; Batch = 256; Ent = 0.005 },

    # [Final Exam] 終極全能 (180 files)
    # 這是最終產出的模型
    @{ Name = "mix_all"; Folder = "mix_all"; Steps = 200000; NSteps = 4096; Batch = 512; Ent = 0.002 }
)

# --- 函數定義 ---
function Find-Model($outDir) {
    $p = Join-Path $outDir "multi_cell_model.zip"
    if (Test-Path $p) { return $p }
    return $null
}

function Train-Stage($st, $resumeFrom) {
    $trainDir = Join-Path $DataRoot "$($st.Folder)\train"
    $outDir = Join-Path $RunsDir $st.Name
    
    Write-Host "`n=== 階段: $($st.Name) ===" -ForegroundColor Yellow
    Write-Host "資料夾: $trainDir"
    
    if (!(Test-Path $trainDir)) { throw "找不到資料夾: $trainDir (請先執行 prepare_data.ps1)" }
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
        # 獎勵設定: 鼓勵連接 (Share)，輕罰斷開 (Break)
        "--w-share", "5.0", "--w-break", "2.0", "--w-hpwl", "0.5", "--w-dummy", "1.0"
    )

    if ($resumeFrom) {
        Write-Host " -> 接續模型: $resumeFrom" -ForegroundColor Gray
        $argsList += "--resume-from"
        $argsList += $resumeFrom
    }
    else {
        Write-Host " -> 從頭開始 (含 BC 預訓練)" -ForegroundColor Gray
    }

    $p = Start-Process -FilePath $Py -ArgumentList $argsList -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "訓練失敗 ExitCode: $($p.ExitCode)" }

    return Find-Model $outDir
}

# --- 主程式 ---
if (!(Test-Path $RunsDir)) { New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null }
$prevModel = $null

foreach ($st in $Stages) {
    try {
        $prevModel = Train-Stage $st $prevModel
        if (!$prevModel) { Write-Warning "未產生模型，停止流程。"; break }
    }
    catch {
        Write-Error $_; break
    }
}
Write-Host "`n全流程結束！最終模型: $prevModel" -ForegroundColor Green