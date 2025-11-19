Param(
    [int]$Nodes = 2500,
    [int]$Seeds = 2,
    [int]$Shards = 2,
    [string]$Topologies = "ws",
    [string]$OzIntensityGrid = "0.8,1.0,1.2",
    [string]$VfGrid = "0.9,1.0",
    [string]$MutationThresholds = "1.2",
    [string]$OutDir = "results",
    [string]$MergedOut = "results/merged.jsonl",
    [string]$PythonExe = "python",
    [switch]$Quiet,
    [switch]$Synchronous,
    [string]$LogDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Prefer workspace virtual environment Python if available
try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
} catch {
    $repoRoot = $PSScriptRoot
}
$venvPy = Join-Path $repoRoot "test-env\Scripts\python.exe"
if ((-not $PSBoundParameters.ContainsKey('PythonExe') -or $PythonExe -eq 'python') -and (Test-Path $venvPy)) {
    $PythonExe = $venvPy
}

if ($Shards -lt 1) { throw "Shards must be >= 1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logTarget = $(if ($LogDir) { $LogDir } else { $OutDir })
New-Item -ItemType Directory -Force -Path $logTarget | Out-Null

# Split OZ intensity grid across shards
$ozList = $OzIntensityGrid.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
if ($ozList.Count -lt $Shards) {
    Write-Host "[WARN] Fewer OZ values than shards; some shards will be empty"
}

function Split-IntoChunks([object[]]$arr, [int]$n) {
    $chunks = @()
    for ($i = 0; $i -lt $n; $i++) { $chunks += ,@() }
    for ($i = 0; $i -lt $arr.Count; $i++) {
        $chunks[$i % $n] += $arr[$i]
    }
    return $chunks
}

$chunks = Split-IntoChunks -arr $ozList -n $Shards

# Absolute path to the bifurcation CLI to avoid CWD issues
$benchPath = Join-Path $repoRoot "benchmarks\bifurcation_landscape.py"

$jobs = @()
for ($i = 0; $i -lt $Shards; $i++) {
    $chunk = $chunks[$i]
    if (-not $chunk -or $chunk.Count -eq 0) { continue }
    $ozArg = ($chunk -join ',')
    $outfile = Join-Path $OutDir ("shard_{0}.jsonl" -f $i)

    $pyArgs = @(
        $benchPath,
        "--nodes", $Nodes,
        "--seeds", $Seeds,
        "--topologies", $Topologies,
        "--oz-intensity-grid", $ozArg,
        "--vf-grid", $VfGrid,
        "--mutation-thresholds", $MutationThresholds
    )
    if ($Quiet) { $pyArgs += "--quiet" }

    Write-Host ("[Shard {0}/{1}] OZ: {2} -> {3}" -f ($i+1), $Shards, $ozArg, $outfile)

    if ($Synchronous) {
        # Run in-process for visibility and robustness
        $env:PYTHONUNBUFFERED = "1"
        $srcPath = Join-Path $PSScriptRoot "..\src"
        if (Test-Path $srcPath) {
            if ($env:PYTHONPATH) { $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH" } else { $env:PYTHONPATH = $srcPath }
        }
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $PythonExe
        $psi.Arguments = ([string]::Join(' ', ($pyArgs | ForEach-Object {
            if ($_ -match '"|\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
        })))
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        if (Test-Path $repoRoot) { $psi.WorkingDirectory = $repoRoot }
        $p = [System.Diagnostics.Process]::Start($psi)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()
        $stdoutPath = Join-Path $logTarget ("shard_{0}.stdout.txt" -f $i)
        $stderrPath = Join-Path $logTarget ("shard_{0}.stderr.txt" -f $i)
        Set-Content -Path $stdoutPath -Value $stdout -Encoding utf8
        Set-Content -Path $stderrPath -Value $stderr -Encoding utf8
        if ($p.ExitCode -ne 0) {
            throw "Shard failed (exit $($p.ExitCode)). See: $stderrPath"
        }
        $lines = $stdout -split "`r?`n" | Where-Object { $_ -match '^\s*\{.*\}\s*$' }
        Set-Content -Path $outfile -Value $lines -Encoding utf8
        Write-Host ("[Shard sync] Wrote {0}" -f $outfile)
    } else {
        $jobs += Start-Job -ScriptBlock {
            Param($pyArgs, $outFile, $pyExe, $scriptDir, $repoRoot, $logTarget)
            $env:PYTHONUNBUFFERED = "1"
            # Ensure TNFR src is importable
            $srcPath = Join-Path $scriptDir "..\src"
            if (Test-Path $srcPath) {
                if ($env:PYTHONPATH) { $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH" } else { $env:PYTHONPATH = $srcPath }
            }
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $pyExe
            $psi.Arguments = ([string]::Join(' ', ($pyArgs | ForEach-Object {
                if ($_ -match '"|\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
            })))
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            # Ensure relative paths in the Python script resolve from repo root
            if (Test-Path $repoRoot) { $psi.WorkingDirectory = $repoRoot }
            $p = [System.Diagnostics.Process]::Start($psi)
            $stdout = $p.StandardOutput.ReadToEnd()
            $stderr = $p.StandardError.ReadToEnd()
            $p.WaitForExit()
            $stdoutPath = Join-Path $logTarget ("shard_{0}.stdout.txt" -f ([int]($outFile -replace '.*shard_(\d+)\.jsonl', '$1')))
            $stderrPath = Join-Path $logTarget ("shard_{0}.stderr.txt" -f ([int]($outFile -replace '.*shard_(\d+)\.jsonl', '$1')))
            Set-Content -Path $stdoutPath -Value $stdout -Encoding utf8
            Set-Content -Path $stderrPath -Value $stderr -Encoding utf8
            if ($p.ExitCode -ne 0) {
                throw "Shard failed: $stderr"
            }
            # Write JSONL lines only
            $lines = $stdout -split "`r?`n" | Where-Object { $_ -match '^\s*\{.*\}\s*$' }
            Set-Content -Path $outFile -Value $lines -Encoding utf8
            Write-Host ("[Shard] Wrote {0}" -f $outFile)
        } -ArgumentList ($pyArgs, $outfile, $PythonExe, $PSScriptRoot, $repoRoot, $logTarget)
    }
}
if (-not $Synchronous) {
    if ($jobs.Count -gt 0) {
        Wait-Job -Job $jobs | Out-Null
        Receive-Job -Job $jobs -ErrorAction Stop | Out-Null
        Remove-Job -Job $jobs -Force
    }
}

# Merge outputs
$mergedDir = Split-Path -Parent $MergedOut
if ($mergedDir) { New-Item -ItemType Directory -Force -Path $mergedDir | Out-Null }
Get-ChildItem (Join-Path $OutDir 'shard_*.jsonl') -ErrorAction SilentlyContinue |
  ForEach-Object { Get-Content $_ } |
  Out-File -Encoding utf8 $MergedOut

Write-Host "[OK] Merged JSONL written to $MergedOut"
