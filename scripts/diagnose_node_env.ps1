param(
    [string]$TestUrl = "",
    [switch]$VerboseOut
)

Write-Host "=== Node.js environment diagnostics ==="

# Basic versions
try {
    $nodeVersion = node -v 2>$null
} catch {}
try {
    $npmVersion = npm -v 2>$null
} catch {}

if ($nodeVersion) { Write-Host "Node: $nodeVersion" } else { Write-Host "Node: not found in PATH" }
if ($npmVersion) { Write-Host "npm:  $npmVersion" } else { Write-Host "npm:  not found in PATH" }

# Key env vars
$proxyVars = @("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","NO_PROXY","no_proxy")
foreach ($v in $proxyVars) {
    try {
        $val = [System.Environment]::GetEnvironmentVariable($v)
        if ($val) { Write-Host "$v=$val" }
    } catch {}
}

# Print sqlite experimental warning hint if Node >= 22
$nodeMajor = 0
if ($nodeVersion -match "v(\d+)") { $nodeMajor = [int]$matches[1] }
if ($nodeMajor -ge 22) {
    Write-Host "Detected Node >=22 (SQLite experimental warnings expected)."
}

# Create a temp JS file to introspect fetch/undici and WebSocket
$js = @'
(async () => {
  const info = {
    versions: process.versions,
    hasGlobalFetch: typeof globalThis.fetch === 'function',
    hasGlobalWebSocket: typeof globalThis.WebSocket === 'function',
    undiciVersion: null,
  };
  try {
    const undici = await import('undici').catch(() => null);
    info.undiciVersion = undici?.default?.Dispatcher?.name ? 'present' : (undici ? 'present' : null);
  } catch {}
  console.log(JSON.stringify(info));
})();
'@

$tmp = Join-Path $env:TEMP "node_env_check_$(Get-Random).mjs"
$js | Out-File -FilePath $tmp -Encoding UTF8 -Force

try {
    $json = node --experimental-default-type=module $tmp 2>$null
    if ($json) {
        Write-Host "Runtime: $json"
    } else {
        Write-Host "Runtime: unable to collect details (node execution failed)"
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp
}

# Optional network check without external calls by default
if ($TestUrl) {
    Write-Host "Testing fetch to: $TestUrl"
    $code = @"
(async () => {
  try {
    const res = await fetch('$TestUrl', { method: 'HEAD' });
    console.log(JSON.stringify({ ok: res.ok, status: res.status }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: String(e) }));
  }
})();
"@
    $tmp2 = Join-Path $env:TEMP "node_fetch_test_$(Get-Random).mjs"
    $code | Out-File -FilePath $tmp2 -Encoding UTF8 -Force
    try {
        $out = node --experimental-default-type=module $tmp2 2>$null
        Write-Host "Fetch result: $out"
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp2
    }
}

Write-Host "=== Done ==="
