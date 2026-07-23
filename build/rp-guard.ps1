# Resource-pack guard for a Minecraft instance (Windows port of rp-guard.sh).
#
# Minecraft silently rewrites options.txt WITHOUT the enabled resource packs (resourcePacks:[]) when a
# resource reload is interrupted (a force-close mid-load, an OOM on a big pack, rapid relaunches). On this
# pack that means losing the custom title, the title background, and the ambient-sound pack all at once.
#
# Run BEFORE every launch (wire rp-guard.bat as the Prism pre-launch command):
#   - If options.txt has a NON-empty resourcePacks list, it is saved as the known-good baseline.
#   - If options.txt has an EMPTY (or missing) resourcePacks list and a baseline exists, it is restored.
# Idempotent, no external deps. Safe to run by hand: powershell -File rp-guard.ps1 <minecraft_dir>
param([Parameter(Mandatory=$true)][string]$Mc)

$opts = Join-Path $Mc "options.txt"
$base = Join-Path $Mc ".hearthlight-rp-baseline"
if (-not (Test-Path $opts)) { Write-Output "rp-guard: no options.txt at $opts (skipping)"; exit 0 }

$lines = Get-Content $opts
$cur = $lines | Where-Object { $_ -match '^resourcePacks:' } | Select-Object -First 1

$isEmpty = (-not $cur) -or ($cur -eq 'resourcePacks:[]') -or ($cur -eq 'resourcePacks:[ ]')

if ($isEmpty) {
    if (Test-Path $base) {
        $good = (Get-Content $base -Raw).TrimEnd("`r", "`n")
        Copy-Item $opts "$opts.bak-rpguard" -Force
        if ($cur) {
            $lines = $lines | ForEach-Object { if ($_ -match '^resourcePacks:') { $good } else { $_ } }
        } else {
            $lines = @($lines) + $good
        }
        Set-Content -Path $opts -Value $lines
        Write-Output "rp-guard: resourcePacks was empty; RESTORED from baseline -> $good"
    } else {
        Write-Output "rp-guard: resourcePacks empty and no baseline yet; nothing to restore (enable your packs once, then this saves them)"
    }
} else {
    Set-Content -Path $base -Value $cur
    Write-Output "rp-guard: baseline up to date -> $cur"
}

# ---- KEYBIND GUARD (per-key engine, mirrors rp-guard.sh) ----------------------------------------------
# Minecraft only writes options.txt on a CLEAN quit, and a session where a mod failed to load DROPS that
# mod's key_ lines; the next clean quit recreates them at the MOD default. Per key, every launch:
#   - a baseline key missing from options.txt is put back (the dropped-line case)
#   - a MONITORED key (mod-defaults list) sitting at its mod default while the baseline says otherwise is
#     a reversion and the baseline wins; two or more at once restores the whole key block from baseline
#   - any other difference is a deliberate rebind and becomes the new baseline
$keybase = Join-Path $Mc ".hearthlight-keybind-baseline"
$deffile = Join-Path $Mc ".hearthlight-mod-defaults"
if (-not (Test-Path $deffile)) { $deffile = Join-Path $PSScriptRoot "keybinds/mod-defaults.txt" }

function Split-Bind($l) { $i = $l.IndexOf(':'); ,@($l.Substring(0, $i), $l.Substring($i + 1)) }

$lines = Get-Content $opts
$keyLines = @($lines | Where-Object { $_ -match '^key_' })
if (-not $keyLines) { exit 0 }

if (-not (Test-Path $keybase)) {
    Set-Content -Path $keybase -Value $keyLines
    Write-Output ("rp-guard: keybind baseline created ({0} binds)" -f $keyLines.Count)
} else {
    $base = [ordered]@{}; foreach ($l in (Get-Content $keybase | Where-Object { $_ -match '^key_' })) { $p = Split-Bind $l; $base[$p[0]] = $p[1] }
    $defs = @{}; if (Test-Path $deffile) { foreach ($l in (Get-Content $deffile | Where-Object { $_ -match '^key_' })) { $p = Split-Bind $l; $defs[$p[0]] = $p[1] } }
    $cur = [ordered]@{}; foreach ($l in $keyLines) { $p = Split-Bind $l; $cur[$p[0]] = $p[1] }

    $reverted = @($cur.Keys | Where-Object { $defs.Contains($_) -and $base.Contains($_) -and $cur[$_] -eq $defs[$_] -and $base[$_] -ne $cur[$_] })
    $missing = @($base.Keys | Where-Object { -not $cur.Contains($_) })

    if ($reverted.Count -ge 2) {
        Copy-Item $opts "$opts.bak-keyguard" -Force
        $block = @($base.Keys | ForEach-Object { "$($_):$($base[$_])" })
        $out = New-Object System.Collections.Generic.List[string]; $done = $false
        foreach ($l in $lines) {
            if ($l -match '^key_') { if (-not $done) { $block | ForEach-Object { $out.Add($_) }; $done = $true } }
            else { $out.Add($l) }
        }
        Set-Content -Path $opts -Value $out
        Write-Output ("rp-guard: {0} monitored binds reverted to mod defaults at once; RESTORED the whole key block from baseline" -f $reverted.Count)
    } elseif ($reverted.Count -eq 1 -or $missing.Count -gt 0) {
        Copy-Item $opts "$opts.bak-keyguard" -Force
        foreach ($k in $reverted) { $cur[$k] = $base[$k] }
        $block = @($cur.Keys | ForEach-Object { "$($_):$($cur[$_])" }) + @($missing | ForEach-Object { "$($_):$($base[$_])" })
        $out = New-Object System.Collections.Generic.List[string]; $done = $false
        foreach ($l in $lines) {
            if ($l -match '^key_') { if (-not $done) { $block | ForEach-Object { $out.Add($_) }; $done = $true } }
            else { $out.Add($l) }
        }
        Set-Content -Path $opts -Value $out
        Write-Output ("rp-guard: repaired keybinds (reverted-to-default restored: {0}, dropped lines put back: {1})" -f $reverted.Count, $missing.Count)
    } else {
        Set-Content -Path $keybase -Value $keyLines
        Write-Output ("rp-guard: keybind baseline up to date ({0} binds)" -f $keyLines.Count)
    }

    # collision watch: JEI binds only fire over items in GUIs and the tag ping deliberately overlaps
    # pick-block, so those never count.
    $seen = @{}
    foreach ($l in (Get-Content $opts | Where-Object { $_ -match '^key_' })) {
        $p = Split-Bind $l
        if ($p[0] -match '^key_key\.jei\.' -or $p[0] -match '^key_key\.jade\.' -or $p[0] -match '^key_key\.sophisticatedcore\.' -or $p[0] -eq 'key_key.hearthlighthaunt.tag_location') { continue }
        if ($p[1] -match 'unknown') { continue }
        if ($seen.Contains($p[1])) { Write-Output ("rp-guard: WARNING both {0} and {1} are bound to {2}" -f $seen[$p[1]], $p[0], $p[1]) }
        else { $seen[$p[1]] = $p[0] }
    }
}
