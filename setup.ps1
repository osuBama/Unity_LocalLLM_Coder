<#
.SYNOPSIS
    One-shot setup: Python env, GPU roles, pinned Ollama instances, models, context fitting,
    config.yaml, orchestrator, WSL networking, OpenClaw provider and (optionally) autostart.

.DESCRIPTION
    Safe to re-run. Every step explains what it will do; anything outside this folder
    (Ollama autostart, .wslconfig, OpenClaw config, scheduled tasks) is only changed after
    you confirm, and is backed up first. A full transcript goes to logs\setup-<time>.log.

.EXAMPLE
    .\setup.ps1                                   # interactive, recommended
    .\setup.ps1 -Yes                              # accept every default
    .\setup.ps1 -PrimaryModel qwen3:14b -MemoryModel qwen3:4b -WslDistro Ubuntu
    .\setup.ps1 -SkipOpenClaw -SkipWsl            # orchestrator only
#>
[CmdletBinding()]
param(
    [string]$PrimaryModel = "",
    [string]$MemoryModel = "",
    [string]$EmbeddingModel = "nomic-embed-text",   # "" or "none" disables semantic search/history recall
    [int]$PrimaryContext = 0,          # starting point for fitting; 0 = from the VRAM plan
    [int]$MemoryContext = 0,
    [string]$PrimaryGpu = "",          # GPU UUIDs (see: nvidia-smi -L); empty = larger VRAM is primary
    [string]$MemoryGpu = "",
    [switch]$SingleGpu,                # force one-GPU mode even with two cards
    [string]$ModelsDir = "",
    [string]$WslDistro = "",
    [int]$PrimaryPort = 11434,
    [int]$MemoryPort = 11435,
    [int]$OrchestratorPort = 8000,
    [switch]$Yes,                      # non-interactive: accept defaults
    [switch]$SkipPull,
    [switch]$SkipFit,
    [switch]$SkipWsl,
    [switch]$SkipOpenClaw,
    [switch]$NoStart,                  # don't start the orchestrator at the end
    [switch]$Autostart,                # create logon tasks without asking
    [switch]$NoAutostart
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $Root "logs"
$BackupDir = Join-Path $Root "backups\setup-$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $Utf8NoBom
try { Start-Transcript -Path (Join-Path $LogDir "setup-$Stamp.log") | Out-Null } catch { }

# ------------------------------------------------------------------ helpers
$script:StepNo = 0
$script:Summary = [ordered]@{}
function Step([string]$title) {
    $script:StepNo++
    Write-Host ""
    Write-Host ("[{0}] {1}" -f $script:StepNo, $title) -ForegroundColor Cyan
}
function Ok([string]$m)   { Write-Host "    OK   $m" -ForegroundColor Green }
function Info([string]$m) { Write-Host "         $m" }
function Warn([string]$m) { Write-Host "    WARN $m" -ForegroundColor Yellow }
function Fail([string]$m) {
    Write-Host "    FAIL $m" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}
function Ask([string]$question, [bool]$default = $true) {
    if ($Yes) { return $default }
    $hint = if ($default) { "[Y/n]" } else { "[y/N]" }
    $a = Read-Host "    $question $hint"
    if ([string]::IsNullOrWhiteSpace($a)) { return $default }
    return $a.Trim().ToLower().StartsWith("y")
}
function AskValue([string]$question, [string]$default) {
    if ($Yes) { return $default }
    $a = Read-Host "    $question [$default]"
    if ([string]::IsNullOrWhiteSpace($a)) { return $default }
    return $a.Trim()
}
function Backup-File([string]$path) {
    if (Test-Path $path) {
        New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
        $dest = Join-Path $BackupDir ((Split-Path $path -Leaf) + ".bak")
        Copy-Item $path $dest -Force
        return $dest
    }
    return $null
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
function Tool([string[]]$toolArgs) {
    $env:PYTHONPATH = $Root
    $out = & $Py -m app.setup_tools @toolArgs
    $line = @($out | Where-Object { $_ -match '^\s*\{' }) | Select-Object -Last 1
    if (-not $line) { Fail "setup helper returned nothing for: $($toolArgs -join ' ')" }
    return ($line | ConvertFrom-Json)
}
function Wait-Http([string]$url, [int]$seconds = 60) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try { return Invoke-RestMethod $url -TimeoutSec 3 } catch { Start-Sleep -Seconds 1 }
    }
    return $null
}
function Quote-Args([string[]]$items) {
    # Start-Process joins arguments with spaces and does not quote them: do it ourselves.
    return (($items | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' ')
}
function Run-PS([string]$file, [string[]]$scriptArgs) {
    $argLine = Quote-Args (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $file) + $scriptArgs)
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $argLine -NoNewWindow -Wait -PassThru
    return $p.ExitCode
}
function Wsl-Bash([string]$distro, [string]$script) {
    # The script goes in over stdin to a login shell (so npm-global / nvm paths for openclaw
    # are loaded). Nothing passes through Windows argument quoting, which mangles double quotes.
    $body = ($script -replace "`r", "") + "`nexit `$? # end`n"
    $out = $body | & wsl.exe -d $distro -- bash -l -s 2>&1
    return @{ Code = $LASTEXITCODE; Out = (($out | ForEach-Object { "$_" }) -join "`n") }
}
function Heredoc([string]$path, [string]$content) {
    return "cat > $path <<'AI_SETUP_EOF'`n$content`nAI_SETUP_EOF"
}
function ToolRaw([string[]]$toolArgs) {
    $env:PYTHONPATH = $Root
    $out = & $Py -m app.setup_tools @toolArgs
    return (@($out | Where-Object { $_ -match '^\s*\{' }) | Select-Object -Last 1)
}
function Wsl-Distros {
    # Newer WSL honours WSL_UTF8=1; older versions write UTF-16 anyway, which reads as
    # characters interleaved with NULs - stripping the NULs handles both.
    $prevEnv = $env:WSL_UTF8
    $env:WSL_UTF8 = "1"
    try { $list = & wsl.exe -l -q 2>$null } finally { $env:WSL_UTF8 = $prevEnv }
    return @($list | ForEach-Object { ($_ -replace "`0", "").Trim() } |
             Where-Object { $_ -and $_ -notlike "docker-desktop*" })
}

Write-Host "Local AI orchestrator setup  ($Root)" -ForegroundColor White
Write-Host "Transcript: logs\setup-$Stamp.log"

# ------------------------------------------------------------- 1. preflight
Step "Preflight: Windows, Python, dependencies"
if (-not (Test-Path (Join-Path $Root "app\api.py")) -or -not (Test-Path (Join-Path $Root "config\config.yaml"))) {
    Fail "Run setup.ps1 from the project folder (app\ and config\ must be next to it)."
}
$build = [Environment]::OSVersion.Version.Build
Ok "Windows build $build"

$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyCmd) { $pyCmd = Get-Command python -ErrorAction SilentlyContinue }
if (-not $pyCmd) { Fail "Python 3.11+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')." }
if ($pyCmd.Name -like "py*" -and $pyCmd.Name -notlike "python*") { $verOut = & $pyCmd.Source -3 --version 2>&1 }
else { $verOut = & $pyCmd.Source --version 2>&1 }
if ("$verOut" -notmatch "Python (\d+)\.(\d+)") { Fail "Could not read the Python version ($verOut)." }
if ([int]$Matches[1] -lt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -lt 11)) {
    Fail "$verOut found; Python 3.11 or newer is required."
}
Ok "$verOut"
if ((Run-PS (Join-Path $Root "scripts\start-orchestrator.ps1") @("-Install")) -ne 0) {
    Fail "Dependency installation failed (see output above)."
}
if (-not (Test-Path $Py)) { Fail "Virtual environment not created at .venv" }
Ok "Virtual environment and dependencies ready"

# ------------------------------------------------------------------ 2. GPUs
Step "GPUs"
$gArgs = @("detect-gpus")
if ($PrimaryGpu) { $gArgs += @("--primary-uuid", $PrimaryGpu) }
if ($MemoryGpu)  { $gArgs += @("--memory-uuid", $MemoryGpu) }
$det = Tool $gArgs
if (-not $det.ok) { Fail "No NVIDIA GPU detected: $($det.error). (AMD/Linux: see docs\DOCUMENTATION.md section 4.)" }
foreach ($g in $det.gpus) { Info ("{0}  {1,-34} {2,5} GB  {3}" -f $g.index, $g.name, $g.vram_gb, $g.uuid) }
$mode = $det.roles.mode
if ($SingleGpu) { $mode = "single" }
$pg = $det.roles.primary
$mg = if ($mode -eq "single") { $pg } else { $det.roles.memory }
if ($mode -eq "dual") {
    Ok "Primary: $($pg.name) ($($pg.vram_gb) GB)   Memory: $($mg.name) ($($mg.vram_gb) GB)"
} else {
    Warn "Single-GPU mode on $($pg.name): both models share it, no isolation (docs\DOCUMENTATION.md 4.4)."
}
$script:Summary["GPU mode"] = $mode

# ---------------------------------------------------------------- 3. Ollama
Step "Ollama"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) { Fail "Ollama not found on PATH. Install it from https://ollama.com/download and re-run." }
$ov = (& ollama --version 2>&1 | Out-String).Trim()
Ok ($ov -replace "\s+", " ")

$startup = [Environment]::GetFolderPath("Startup")
$links = @(Get-ChildItem -Path $startup -Filter "*ollama*.lnk" -ErrorAction SilentlyContinue)
$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$runVals = @()
if (Test-Path $runKey) {
    $props = Get-ItemProperty $runKey
    $runVals = @($props.PSObject.Properties | Where-Object { $_.Name -notlike "PS*" -and "$($_.Value)" -match "ollama" })
}
if ($links.Count -or $runVals.Count) {
    Info "The Ollama tray app starts an UNPINNED server on port $PrimaryPort at logon, which would grab both GPUs."
    if (Ask "Disable the Ollama tray app's autostart? (backed up, reversible)" $true) {
        New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
        foreach ($l in $links) { Move-Item $l.FullName (Join-Path $BackupDir $l.Name) -Force; Ok "Moved $($l.Name) to $BackupDir" }
        foreach ($v in $runVals) {
            "$($v.Name)=$($v.Value)" | Set-Content (Join-Path $BackupDir "run-key-$($v.Name).txt")
            Remove-ItemProperty -Path $runKey -Name $v.Name
            Ok "Removed Run entry '$($v.Name)' (value saved in $BackupDir)"
        }
    } else { Warn "Left autostart in place: after a reboot, quit the tray app before starting the pinned instances." }
} else { Ok "No Ollama autostart entry found" }

$running = @(Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue)
if ($running.Count) {
    if (Ask "Stop the $($running.Count) running Ollama process(es) so the pinned instances can start?" $true) {
        $running | Stop-Process -Force
        Start-Sleep -Seconds 2
        Ok "Stopped"
    } else { Fail "Ports must be free for the pinned instances. Quit Ollama and re-run." }
}

if (-not $ModelsDir) {
    $ModelsDir = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
    if (-not $ModelsDir) { $ModelsDir = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "Machine") }
    if (-not $ModelsDir) { $ModelsDir = Join-Path $env:USERPROFILE ".ollama\models" }
    $ModelsDir = AskValue "Models folder (shared by both instances)" $ModelsDir
}
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Ok "Models folder: $ModelsDir"

# ------------------------------------------------------------------ 4. plan
Step "Models"
$desktopGb = [math]::Round($pg.used_mib / 1024, 1)
$planArgs = @("plan", "--primary-gb", "$($pg.vram_gb)", "--desktop-gb", "$desktopGb")
if ($mode -eq "dual") { $planArgs += @("--memory-gb", "$($mg.vram_gb)") }
$plan = Tool $planArgs
Info "Suggested from VRAM ($desktopGb GB already in use on the primary):"
Info "  primary $($plan.primary_model) @ $($plan.primary_ctx) ctx    memory $($plan.memory_model) @ $($plan.memory_ctx) ctx"
if (-not $PrimaryModel) { $PrimaryModel = AskValue "Primary model" $plan.primary_model }
if (-not $MemoryModel)  { $MemoryModel  = AskValue "Memory model"  $plan.memory_model }
if ($EmbeddingModel -eq "none") { $EmbeddingModel = "" }
if (-not $PrimaryContext) { $PrimaryContext = [int]$plan.primary_ctx }
if (-not $MemoryContext)  { $MemoryContext  = [int]$plan.memory_ctx }
Ok "Primary $PrimaryModel   Memory $MemoryModel   Embeddings $(if ($EmbeddingModel) { $EmbeddingModel } else { 'off' })"
$script:Summary["Models"] = "$PrimaryModel / $MemoryModel$(if ($EmbeddingModel) { " / $EmbeddingModel" })"

# -------------------------------------------------------- 5. pinned instances
Step "Pinned Ollama instances"
$memPort = if ($mode -eq "single") { $PrimaryPort } else { $MemoryPort }
$instArgs = @("-PrimaryGpu", $pg.uuid, "-PrimaryPort", "$PrimaryPort", "-MemoryPort", "$MemoryPort",
              "-PrimaryContext", "$PrimaryContext", "-MemoryContext", "$MemoryContext",
              "-ModelsDir", $ModelsDir, "-LogDir", $LogDir, "-Force")
if ($mode -eq "single") { $instArgs += "-SingleGpu" } else { $instArgs += @("-MemoryGpu", $mg.uuid) }
$instScript = Join-Path $Root "scripts\start-ollama-instances.ps1"
Run-PS $instScript @("-Stop", "-LogDir", $LogDir) | Out-Null
if ((Run-PS $instScript $instArgs) -ne 0) { Fail "Could not start the Ollama instances (see logs\ollama-*.log)." }
foreach ($port in @($PrimaryPort, $memPort) | Select-Object -Unique) {
    $v = Wait-Http "http://127.0.0.1:$port/api/version" 60
    if (-not $v) { Fail "Ollama on port $port did not come up (see logs\ollama-*.log)." }
    Ok "127.0.0.1:$port  Ollama $($v.version)"
}

# ------------------------------------------------------------------- 6. pull
Step "Models download"
if ($SkipPull) { Info "Skipped (-SkipPull)" }
else {
    foreach ($m in @($PrimaryModel, $MemoryModel, $EmbeddingModel) | Where-Object { $_ } | Select-Object -Unique) {
        $env:OLLAMA_HOST = "127.0.0.1:$PrimaryPort"
        & ollama pull $m
        $code = $LASTEXITCODE
        Remove-Item Env:OLLAMA_HOST -ErrorAction SilentlyContinue
        if ($code -ne 0) { Fail "ollama pull $m failed. Check the model name at https://ollama.com/library" }
        Ok "$m"
    }
}

# --------------------------------------------------------------- 7. fit ctx
Step "Context sizes (fit 100% on GPU)"
if ($SkipFit) {
    Info "Skipped (-SkipFit): using $PrimaryContext / $MemoryContext"
} else {
    Info "Loading each model and stepping num_ctx down until it sits fully on its GPU..."
    $mfArgs = @("fit-context", "--base-url", "http://127.0.0.1:$memPort", "--model", $MemoryModel,
                "--start", "$MemoryContext")
    if ($EmbeddingModel) { $mfArgs += @("--embed-model", $EmbeddingModel) }   # shares the memory GPU
    $mfit = Tool $mfArgs
    if (-not $mfit.ok) { Fail "$($mfit.error) (re-run with -MemoryModel <smaller>)" }
    $MemoryContext = [int]$mfit.num_ctx
    Ok "$MemoryModel  num_ctx $MemoryContext  ($($mfit.vram_gb) GB)"
    $pfArgs = @("fit-context", "--base-url", "http://127.0.0.1:$PrimaryPort", "--model", $PrimaryModel,
                "--start", "$PrimaryContext")
    if ($mode -eq "single") {
        $pfArgs += @("--keep", $MemoryModel)
        if ($EmbeddingModel) { $pfArgs += @("--keep", $EmbeddingModel) }
    }
    $pfit = Tool $pfArgs
    if (-not $pfit.ok) { Fail "$($pfit.error) (re-run with -PrimaryModel <smaller>)" }
    $PrimaryContext = [int]$pfit.num_ctx
    Ok "$PrimaryModel  num_ctx $PrimaryContext  ($($pfit.vram_gb) GB)"
}
$script:Summary["Context"] = "primary $PrimaryContext / memory $MemoryContext"

# ------------------------------------------------------------- 8. config.yaml
Step "config.yaml"
$bud = Tool @("budgets", "--ctx", "$PrimaryContext")
$updates = [ordered]@{
    "paths.root"                    = $Root
    "ollama.primary.base_url"       = "http://127.0.0.1:$PrimaryPort"
    "ollama.primary.model"          = $PrimaryModel
    "ollama.primary.num_ctx"        = $PrimaryContext
    "ollama.memory.base_url"        = "http://127.0.0.1:$memPort"
    "ollama.memory.model"           = $MemoryModel
    "ollama.memory.num_ctx"         = $MemoryContext
    "memory.max_context_tokens"     = [int]$bud.memory_budget
    "stable_memory.max_tokens"      = [int]$bud.memory_base_budget
    "embeddings.enabled"            = [bool]$EmbeddingModel
    "embeddings.model"              = $(if ($EmbeddingModel) { $EmbeddingModel } else { "nomic-embed-text" })
    "application.host"              = "127.0.0.1"
    "application.port"              = $OrchestratorPort
}
$updFile = Join-Path $env:TEMP "ai-setup-$Stamp.json"
[IO.File]::WriteAllText($updFile, ($updates | ConvertTo-Json), $Utf8NoBom)
$res = Tool @("set-config", (Join-Path $Root "config\config.yaml"), "--json-file", $updFile)
Remove-Item $updFile -ErrorAction SilentlyContinue
if (-not $res.ok) { Fail "config.yaml not updated: $($res.error)" }
Ok "Updated (backup: $(Split-Path $res.backup -Leaf)); memory budget $($bud.memory_budget), base $($bud.memory_base_budget)"
if (-not (Test-Path (Join-Path $Root "memory\*.md"))) { Info "Memory starts empty." }
else { Info "memory\*.md already has entries; edit or delete them if they describe another setup." }

# --------------------------------------------------------- 9. orchestrator
Step "Orchestrator"
$health = $null
if ($NoStart) { Info "Not started (-NoStart). Start it with scripts\start-orchestrator.ps1" }
else {
    $existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                  Where-Object { $_.ExecutablePath -eq $Py -and $_.CommandLine -match "app\.main|app\.cli serve" })
    if ($existing.Count) {
        if (Ask "An orchestrator is already running. Restart it to load the new config?" $true) {
            $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
        }
    }
    $null = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList (Quote-Args @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "scripts\start-orchestrator.ps1")))
    $health = Wait-Http "http://127.0.0.1:$OrchestratorPort/health" 60
    if (-not $health) { Fail "Orchestrator did not answer on port $OrchestratorPort (see logs\orchestrator.log)." }
    Ok "http://127.0.0.1:$OrchestratorPort  status: $($health.status)"
    foreach ($w in @($health.warnings)) { if ($w) { Warn $w } }
    if ($EmbeddingModel -and (Test-Path (Join-Path $Root "conversations\*.jsonl"))) {
        Info "Existing conversation history can be made searchable (history recall). This embeds every"
        Info "recorded turn on the memory GPU and can take a few minutes for large histories."
        if (Ask "Index existing history now?" $true) {
            $env:PYTHONPATH = $Root
            & $Py -m app.cli memory reindex
            if ($LASTEXITCODE -eq 0) { Ok "History indexed" } else { Warn "Reindex failed; run '.\ai memory reindex' later" }
        }
    }
}

# ------------------------------------------------------------------ 10. WSL
Step "WSL networking"
$wslOk = $false
if ($SkipWsl -and $SkipOpenClaw) { Info "Skipped" }
elseif (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { Warn "WSL not installed; skipping WSL and OpenClaw steps."; $SkipOpenClaw = $true }
elseif ($SkipWsl) { Info "Skipped (-SkipWsl)"; $wslOk = $true }
elseif ($build -lt 22621) {
    Warn "Mirrored networking needs Windows 11 22H2+ (build 22621). OpenClaw in WSL cannot reach 127.0.0.1:$OrchestratorPort."
    Info "Options: upgrade Windows, or run OpenClaw on Windows. See docs\DOCUMENTATION.md 7.3."
} else {
    $wslcfg = Join-Path $env:USERPROFILE ".wslconfig"
    $text = if (Test-Path $wslcfg) { [IO.File]::ReadAllText($wslcfg) } else { "" }
    $lines = New-Object System.Collections.Generic.List[string]
    if ($text) { $text -split "`r?`n" | ForEach-Object { $lines.Add($_) } }
    $sec = -1; for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i].Trim() -match '^\[wsl2\]$') { $sec = $i; break } }
    if ($sec -lt 0) { if ($lines.Count -and $lines[$lines.Count - 1].Trim()) { $lines.Add("") }; $lines.Add("[wsl2]"); $sec = $lines.Count - 1 }
    $end = $lines.Count; for ($i = $sec + 1; $i -lt $lines.Count; $i++) { if ($lines[$i].Trim().StartsWith("[")) { $end = $i; break } }
    $have = @{}
    for ($i = $sec + 1; $i -lt $end; $i++) { if ($lines[$i] -match '^\s*([A-Za-z]+)\s*=\s*(.*?)\s*$') { $have[$matches[1].ToLower()] = $i } }
    $changes = @()
    if ($have.ContainsKey("networkingmode")) {
        if ($lines[$have["networkingmode"]] -notmatch "mirrored") { $lines[$have["networkingmode"]] = "networkingMode=mirrored"; $changes += "networkingMode=mirrored" }
    } else { $lines.Insert($sec + 1, "networkingMode=mirrored"); $end++; $changes += "networkingMode=mirrored" }
    if (-not $have.ContainsKey("memory")) {
        $ramGb = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        $wslMem = [math]::Max(4, [math]::Floor($ramGb * 0.35))
        $lines.Insert($sec + 1, "memory=${wslMem}GB"); $changes += "memory=${wslMem}GB (of $ramGb GB RAM)"
    }
    if ($changes.Count) {
        Info "Proposed .wslconfig changes: $($changes -join ', ')"
        if (Ask "Apply them to $wslcfg? (backed up)" $true) {
            $bk = Backup-File $wslcfg
            [IO.File]::WriteAllText($wslcfg, (($lines -join "`r`n").TrimEnd() + "`r`n"), $Utf8NoBom)
            Ok "Written$(if ($bk) { " (backup: $bk)" })"
            Info "WSL must restart for this: 'wsl --shutdown' closes ALL running distros."
            if (Ask "Run 'wsl --shutdown' now?" $true) { & wsl.exe --shutdown; Start-Sleep -Seconds 3; Ok "WSL restarted on next use" }
            else { Warn "Run 'wsl --shutdown' yourself before using OpenClaw." }
            $wslOk = $true
        }
    } else { Ok ".wslconfig already has mirrored networking"; $wslOk = $true }
}

# -------------------------------------------------------------- 11. OpenClaw
Step "OpenClaw (WSL)"
if ($SkipOpenClaw) { Info "Skipped" }
else {
    $distros = Wsl-Distros
    if ($WslDistro) { $candidates = @($WslDistro) } else { $candidates = $distros }
    $distro = $null
    foreach ($d in $candidates) {
        $r = Wsl-Bash $d 'command -v openclaw >/dev/null 2>&1 && openclaw --version 2>/dev/null | head -1'
        if ($r.Code -eq 0) { $distro = $d; Ok "Found OpenClaw in '$d' ($($r.Out.Trim()))"; break }
    }
    if (-not $distro) {
        Warn "OpenClaw not found in any WSL distro ($($distros -join ', ')). Install it, then re-run with -WslDistro <name>."
    } else {
        $pt = @("openclaw-patch", "--model", $PrimaryModel, "--ctx", "$PrimaryContext",
                "--base-url", "http://127.0.0.1:$OrchestratorPort")
        $patch = ToolRaw $pt
        $provider = ToolRaw ($pt + "--provider-only")
        if (-not $patch -or -not $provider) { Fail "Could not build the OpenClaw patch." }
        Info "Will point OpenClaw's Ollama provider at http://127.0.0.1:$OrchestratorPort and set ollama/$PrimaryModel as default."
        if (Ask "Update the OpenClaw config in '$distro'? (backed up first)" $true) {
            $bk = Wsl-Bash $distro (@(
                'f=$(openclaw config file 2>/dev/null | tail -1)',
                '[ -f "$f" ] || f="$HOME/.openclaw/openclaw.json"',
                ('if [ -f "$f" ]; then cp "$f" "$f.bak-{0}" && echo "$f.bak-{0}"; fi' -f $Stamp)) -join "`n")
            if ($bk.Out.Trim()) { Ok "Backup: $($bk.Out.Trim())" }
            $tmp = "/tmp/ai-setup-$Stamp"
            $applied = $false
            # 1) config patch (current OpenClaw): dry-run validates against its schema first.
            $r = Wsl-Bash $distro (@("mkdir -p $tmp", (Heredoc "$tmp/patch.json" $patch),
                "openclaw config patch --stdin --dry-run < $tmp/patch.json >/dev/null 2>&1 || exit 3",
                "openclaw config patch --stdin < $tmp/patch.json") -join "`n")
            if ($r.Code -eq 0) { $applied = $true; Ok "Patched with 'openclaw config patch'" }
            # 2) config set (older OpenClaw).
            if (-not $applied) {
                $r = Wsl-Bash $distro (@("mkdir -p $tmp", (Heredoc "$tmp/provider.json" $provider),
                    ('openclaw config set models.providers.ollama "$(cat {0}/provider.json)" --strict-json' -f $tmp)) -join "`n")
                if ($r.Code -eq 0) {
                    $applied = $true; Ok "Set with 'openclaw config set'"
                    $mcpJson = '{"url":"http://127.0.0.1:' + $OrchestratorPort + '/mcp","transport":"streamable-http"}'
                    $mr = Wsl-Bash $distro (@("mkdir -p $tmp", (Heredoc "$tmp/mcp.json" $mcpJson),
                        ('openclaw config set mcp.servers.local-docs "$(cat {0}/mcp.json)" --strict-json' -f $tmp)) -join "`n")
                    if ($mr.Code -eq 0) { Ok "Docs tools registered (MCP server local-docs)" }
                    else { Warn "Register the docs tools yourself: .\ai docs mcp shows the snippet" }
                    $ms = Wsl-Bash $distro "openclaw models set ollama/$PrimaryModel"
                    if ($ms.Code -eq 0) { Ok "Default model ollama/$PrimaryModel" }
                    else { Warn "Set the default model yourself: openclaw models set ollama/$PrimaryModel" }
                }
            }
            # 3) leave the snippet for a manual merge.
            if (-not $applied) {
                Wsl-Bash $distro (@('mkdir -p "$HOME/.openclaw"', (Heredoc '"$HOME/.openclaw/orchestrator-provider.json"' $patch)) -join "`n") | Out-Null
                Warn "This OpenClaw version accepted neither 'config patch' nor 'config set':"
                Warn ((@($r.Out.Trim() -split "`n") | Select-Object -Last 2) -join " ")
                Warn "Merge ~/.openclaw/orchestrator-provider.json into your OpenClaw config by hand (docs\DOCUMENTATION.md 7.2)."
            } else {
                $rs = Wsl-Bash $distro 'openclaw gateway restart'
                if ($rs.Code -eq 0) { Ok "Gateway restarted" } else { Warn "Restart the gateway yourself: openclaw gateway restart" }
                if ($health) {
                    $pr = Wsl-Bash $distro 'openclaw mcp probe local-docs 2>&1 | tail -5'
                    if ($pr.Code -eq 0 -and $pr.Out -match "docs_search") { Ok "OpenClaw sees the docs tools (docs_search, docs_lookup)" }
                    else { Info "Could not confirm the docs tools yet; check later with: openclaw mcp probe local-docs" }
                }
            }
            Wsl-Bash $distro "rm -rf $tmp" | Out-Null
            $script:Summary["OpenClaw"] = if ($applied) { "configured in '$distro'" } else { "manual merge needed" }
        }
        if ($health) {
            $t = Wsl-Bash $distro ("if command -v curl >/dev/null; then curl -s -m 5 http://127.0.0.1:$OrchestratorPort/api/tags; " +
                                   "else wget -qO- -T 5 http://127.0.0.1:$OrchestratorPort/api/tags; fi")
            if ($t.Code -eq 0 -and $t.Out -match '"models"') { Ok "WSL reaches the orchestrator at 127.0.0.1:$OrchestratorPort" }
            else { Warn "WSL cannot reach 127.0.0.1:$OrchestratorPort yet (mirrored networking needs 'wsl --shutdown' after the .wslconfig change)." }
        }
    }
}

# ------------------------------------------------------------- 12. autostart
Step "Autostart at logon"
$doAuto = $false
if ($Autostart) { $doAuto = $true } elseif (-not $NoAutostart) { $doAuto = Ask "Start the Ollama instances and the orchestrator automatically at logon?" $false }
if ($doAuto) {
    try {
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $instArgLine = Quote-Args $instArgs
        $a1 = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
            -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$instScript`" $instArgLine"
        Register-ScheduledTask -TaskName "AI Orchestrator - Ollama instances" -Action $a1 -Trigger $trigger `
            -Settings $settings -Description "Pinned Ollama instances (setup.ps1)" -Force | Out-Null
        $trigger2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        $trigger2.Delay = "PT45S"
        $a2 = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
            -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$(Join-Path $Root 'scripts\start-orchestrator.ps1')`""
        Register-ScheduledTask -TaskName "AI Orchestrator - server" -Action $a2 -Trigger $trigger2 `
            -Settings $settings -Description "Orchestrator on 127.0.0.1:$OrchestratorPort (setup.ps1)" -Force | Out-Null
        Ok "Tasks 'AI Orchestrator - Ollama instances' and 'AI Orchestrator - server' registered (Task Scheduler)"
        $script:Summary["Autostart"] = "on"
    } catch { Warn "Could not register scheduled tasks: $($_.Exception.Message)" }
} else { Info "Not configured. Start manually: scripts\start-ollama-instances.ps1 (same arguments), then scripts\start-orchestrator.ps1" }

# ---------------------------------------------------------------- summary
Write-Host ""
Write-Host "Setup complete" -ForegroundColor Green
$script:Summary["Orchestrator"] = "http://127.0.0.1:$OrchestratorPort  (Ollama-compatible; point clients here)"
$script:Summary["Console"] = "http://127.0.0.1:$OrchestratorPort/ui  (or .\ai ui)"
$script:Summary["Ollama"] = if ($mode -eq "single") { "127.0.0.1:$PrimaryPort" } else { "primary 127.0.0.1:$PrimaryPort, memory 127.0.0.1:$MemoryPort" }
$script:Summary["Config"] = "config\config.yaml (backup next to it)"
if (Test-Path $BackupDir) { $script:Summary["Backups"] = $BackupDir }
foreach ($k in $script:Summary.Keys) { Write-Host ("  {0,-13} {1}" -f $k, $script:Summary[$k]) }
Write-Host ""
Write-Host "Next:  .\ai ui    .\ai status    .\ai chat -v    .\ai eval sessions    (docs\DOCUMENTATION.md sections 6, 7, 10, 12)"
Write-Host "Re-run .\setup.ps1 any time; it is safe to repeat. Instance launch arguments are in this log."
try { Stop-Transcript | Out-Null } catch { }
