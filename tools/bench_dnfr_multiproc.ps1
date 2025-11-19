Param(
        [int]$Nodes = 8192,
        [double]$P = 0.05,
        [int]$Repeats = 1,
        [int]$Concurrency = 0,
        [string]$PythonExe = "python",
        [string]$OutDir = "results/dnfr_mp",
        [switch]$Warmup,
        [int]$StaggerSeconds = 0,
        [Alias("Affinity","SetAffinity")][string[]]$CoreSets
)

<#
DNFR Multi-process Benchmark Orchestrator

Parameters:
    -Nodes             : Node count per process (default 8192)
    -P                 : Edge probability per process (default 0.05)
    -Repeats           : Benchmark repeats inside each process
    -Concurrency       : Number of parallel processes (auto if 0)
    -PythonExe         : Python interpreter (auto selects repo venv)
    -OutDir            : Output directory for stdout/stderr logs
    -Warmup            : Performs one reduced repeat warm-up per process
    -StaggerSeconds    : Seconds to sleep between launches to avoid simultaneous memory bandwidth surge
    -CoreSets / -Affinity / -SetAffinity : Per-process core lists (e.g. '0,1' '2,3' '4'...) used to build affinity masks.

Affinity specification:
    Provide an array where each element is a comma-separated list of logical core indices for the corresponding process.
    Example:
        .\bench_dnfr_multiproc.ps1 -Concurrency 3 -CoreSets '0,1','2,3','4'
    If fewer entries than processes are provided, the last entry is reused.
    Mask construction: bitmask = Σ (1 << coreIndex). The resulting mask is assigned to Process.ProcessorAffinity.

Staggering rationale:
    Launching many identical high-memory-init processes simultaneously can force contention at the cache/memory knee.
    A small stagger (e.g. 1–2s) improves aggregate stability of steps/sec without affecting comparability.

Canonicity: Pure orchestration/telemetry — does not alter TNFR physics or operator semantics. All structural metrics remain process-local.
Note: Do not use affinity to bias physics interpretation; it is strictly a performance control.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
} catch { $repoRoot = $PSScriptRoot }

# Default to venv Python if available
$venvPy = Join-Path $repoRoot "test-env\Scripts\python.exe"
if ((-not $PSBoundParameters.ContainsKey('PythonExe') -or $PythonExe -eq 'python') -and (Test-Path $venvPy)) {
    $PythonExe = $venvPy
}

# Determine concurrency if not provided
if ($Concurrency -le 0) {
    try {
        $cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
        if (-not $cores) { $cores = [int]$env:NUMBER_OF_PROCESSORS }
        if (-not $cores) { $cores = 2 }
    } catch { $cores = 2 }
    $Concurrency = [Math]::Min($cores, 4)  # keep it modest by default
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$benchPath = Join-Path $repoRoot "benchmarks\compute_dnfr_benchmark.py"
if (-not (Test-Path $benchPath)) { throw "Benchmark not found: $benchPath" }

$jobs = @()
$logs = @()

Write-Host ("[DNFR-MP] Launching {0} processes: N={1}, p={2}, repeats={3}" -f $Concurrency, $Nodes, $P, $Repeats)
for ($i = 0; $i -lt $Concurrency; $i++) {
    if ($StaggerSeconds -gt 0 -and $i -gt 0) { Start-Sleep -Seconds $StaggerSeconds }
    $stdoutPath = Join-Path $OutDir ("run_{0}.stdout.txt" -f $i)
    $stderrPath = Join-Path $OutDir ("run_{0}.stderr.txt" -f $i)
    $logs += @{ idx = $i; out = $stdoutPath; err = $stderrPath }

    $argList = @(
        $benchPath,
        "--nodes", $Nodes,
        "--edge-probabilities", ([string]::Format("{0}", $P)),
        "--repeats", $Repeats
    )

    $coreSetForThis = $null
    if ($CoreSets -and $CoreSets.Count -gt 0) {
        if ($i -lt $CoreSets.Count) { $coreSetForThis = $CoreSets[$i] } else { $coreSetForThis = $CoreSets[-1] }
    }

    $jobs += Start-Job -ScriptBlock {
        Param($pythonExe, $repoRoot, $argList, $stdoutPath, $stderrPath, $warmup, $nodes, $p, $coreSet)
        $env:PYTHONUNBUFFERED = "1"
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $pythonExe
        $psi.Arguments = ([string]::Join(' ', ($argList | ForEach-Object {
            if ($_ -match '"|\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
        })))
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        if (Test-Path $repoRoot) { $psi.WorkingDirectory = $repoRoot }

        if ($warmup) {
            # Warm-up run (repeat once) to prime imports/JIT; discard output
            $warmArgs = $argList.Clone()
            # Replace repeats value with 1 for warm-up
            for ($wi = 0; $wi -lt $warmArgs.Count; $wi++) { if ($warmArgs[$wi] -eq "--repeats") { $warmArgs[$wi+1] = 1 } }
            $wpsi = New-Object System.Diagnostics.ProcessStartInfo
            $wpsi.FileName = $pythonExe
            $wpsi.Arguments = ([string]::Join(' ', ($warmArgs | ForEach-Object {
                if ($_ -match '"|\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
            })))
            $wpsi.RedirectStandardOutput = $true
            $wpsi.RedirectStandardError = $true
            $wpsi.UseShellExecute = $false
            if (Test-Path $repoRoot) { $wpsi.WorkingDirectory = $repoRoot }
            $wp = [System.Diagnostics.Process]::Start($wpsi)
            $null = $wp.StandardOutput.ReadToEnd(); $null = $wp.StandardError.ReadToEnd(); $wp.WaitForExit()
            # Ignore non-zero warm-up exit silently but surface if main run fails
        }

        $p = [System.Diagnostics.Process]::Start($psi)

        if ($coreSet) {
            try {
                $coreInts = $coreSet -split ',' | ForEach-Object { ($_ -replace '\s','') } | Where-Object { $_ -match '^[0-9]+$' } | ForEach-Object { [int]$_ }
                if ($coreInts.Count -gt 0) {
                    $mask = 0
                    foreach ($c in $coreInts) { $mask = $mask -bor (1 -shl $c) }
                    $p.ProcessorAffinity = [IntPtr]$mask
                }
            } catch {
                Write-Host "[DNFR-MP][WARN] Affinity set failed for '$coreSet': $_" -ForegroundColor Yellow
            }
        }
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()
        Set-Content -Path $stdoutPath -Value $stdout -Encoding utf8
        Set-Content -Path $stderrPath -Value $stderr -Encoding utf8
        if ($p.ExitCode -ne 0) { throw "DNFR run failed (exit $($p.ExitCode)). See $stderrPath" }
    } -ArgumentList ($PythonExe, $repoRoot, $argList, $stdoutPath, $stderrPath, $Warmup.IsPresent, $Nodes, $P, $coreSetForThis)
}

Wait-Job -Job $jobs | Out-Null
Receive-Job -Job $jobs -ErrorAction Stop | Out-Null
Remove-Job -Job $jobs -Force

# Parse results
function Get-VectorizedSeconds([string]$text, [int]$nodes, [double]$p) {
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -match "^\s*${nodes}\s*\|\s*${p}\s*\|") {
            # Split columns by '|', column 4 is the vectorized timings (best/med/mean/worst)
            $cols = ($line -split "\|").Trim()
            if ($cols.Count -ge 4) {
                $vec = $cols[3]
                # Extract first float
                if ($vec -match "([0-9]+\.[0-9]+)") {
                    return [double]$Matches[1]
                }
            }
        }
    }
    return $null
}

$per = @()
foreach ($l in $logs) {
    $txt = (Get-Content $l.out -Raw)
    $sec = Get-VectorizedSeconds -text $txt -nodes $Nodes -p $P
    if ($null -ne $sec -and $sec -gt 0) {
        $per += 1.0 / $sec
    }
}

if ($per.Count -eq 0) {
    Write-Host "[DNFR-MP] No results parsed. Check logs in $OutDir" -ForegroundColor Yellow
    exit 2
}

$total = ($per | Measure-Object -Sum).Sum
$min = ($per | Measure-Object -Minimum).Minimum
$max = ($per | Measure-Object -Maximum).Maximum
$avg = [double]($total / $per.Count)

Write-Host "[DNFR-MP] Per-process steps/sec: " ($per -join ", ")
Write-Host ("[DNFR-MP] Aggregate steps/sec: {0:F2} (min {1:F2}, avg {2:F2}, max {3:F2})" -f $total, $min, $avg, $max)
