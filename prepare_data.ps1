# ========================================================
# Auto-prepare Mix Folders (Corrected Path Version)
# ========================================================
$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
# 修正：移除 out_cells
$BaseDir = Join-Path $Root "preprocess\_clean\bins"

$MixConfig = @{
    "mix_0-9"  = @("0-5", "6-9");
    "mix_0-20" = @("0-5", "6-9", "10-15", "16-20");
    "21-50"    = @("21-25", "26-30", "31-50");
    "mix_all"  = @("0-5", "6-9", "10-15", "16-20", "21-25", "26-30", "31-50")
}

Write-Host "Starting data preparation..." -ForegroundColor Cyan
Write-Host "Base Directory: $BaseDir" -ForegroundColor Gray

foreach ($TargetName in $MixConfig.Keys) {
    $TargetDir = Join-Path $BaseDir "$TargetName\train"
    
    if (Test-Path $TargetDir) {
        Get-ChildItem -Path $TargetDir -Recurse | Remove-Item -Force -Recurse
    }
    else {
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    }
    
    Write-Host "Processing target: $TargetName"

    $SourceBins = $MixConfig[$TargetName]
    $TotalCount = 0
    
    foreach ($Bin in $SourceBins) {
        # 1. Check standard path (with \train)
        $SrcDir = Join-Path $BaseDir "$Bin\train"
        
        # 2. Fallback: Check root bin folder (without \train)
        # 根據您的路徑結構，檔案似乎直接在 bins\31-50 底下
        if (-not (Test-Path $SrcDir)) {
            $SrcDir = Join-Path $BaseDir "$Bin"
        }

        if (Test-Path $SrcDir) {
            $Files = Get-ChildItem -Path $SrcDir -Filter "*.json"
            if ($Files.Count -eq 0) {
                Write-Warning "  Folder exists but EMPTY: $SrcDir"
            }
            foreach ($File in $Files) {
                Copy-Item -Path $File.FullName -Destination $TargetDir -Force
                $TotalCount++
            }
        }
        else {
            Write-Warning "  Source folder NOT FOUND: $SrcDir"
        }
    }
    
    Write-Host "  -> Copied $TotalCount files to $TargetName" -ForegroundColor Green
}

Write-Host "`nData preparation complete!" -ForegroundColor Cyan