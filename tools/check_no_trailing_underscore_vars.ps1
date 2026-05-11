param(
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"

# Heuristic check:
# - Variables (locals, globals, members, statics) MUST NOT end with '_'.
# - Function parameters are allowed to end with '_' (and are enforced separately by clang-tidy).
#
# This script intentionally targets only patterns that look like *variable declarations*
# ending with ';', '=', '{', or ',' to reduce false positives on function parameters.

$extensions = @(".h", ".hpp", ".inl", ".cpp", ".cc", ".cxx", ".cppm")
$roots = @(
    (Join-Path $Root "include"),
    (Join-Path $Root "modules"),
    (Join-Path $Root "test"),
    (Join-Path $Root "bench")
)

$files = @()
foreach ($r in $roots) {
    if (Test-Path $r) {
        $files += Get-ChildItem -Path $r -Recurse -File | Where-Object { $_.Extension -in $extensions }
    }
}

# Match (heuristic): "<type tokens> <identifier_> ;|=|{"
# Notes:
# - Intentionally DOES NOT treat ',' as a terminator, because many parameters are split
#   across lines like "float edge0_," and would be false positives (parameters are allowed).
# - This is a best-effort guardrail; clang-tidy enforces parameter naming separately.
$decl_re = '^\s*(?:alignas\s*\(.*\)\s*)?(?:(?:constexpr|consteval|constinit|inline|static|thread_local|mutable|volatile|register)\s+)*(?=[A-Za-z_])' +
           '(?:[\w:<>]+\s+)*[\w:<>]+\s+([a-z][A-Za-z0-9]*_)\s*(?:=|;|\{)'

$violations = @()
foreach ($f in $files) {
    $i = 0
    foreach ($line in Get-Content -LiteralPath $f.FullName -Encoding utf8) {
        $i++
        if ($line -match $decl_re) {
            if ($line -match '^\s*return\b') {
                continue
            }
            # If the identifier is immediately followed by '(' on the same line,
            # treat it as a function (not a variable). This also filters some macros.
            if ($line -match '\b[a-z][A-Za-z0-9]*_\s*\(') {
                continue
            }
            $violations += [PSCustomObject]@{
                File = $f.FullName
                Line = $i
                Text = $line.TrimEnd()
            }
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Found variables ending with '_' (forbidden by style):"
    foreach ($v in $violations) {
        Write-Host ("{0}:{1}: {2}" -f $v.File, $v.Line, $v.Text)
    }
    exit 1
}

Write-Host "OK: no variable declarations ending with '_' found."
exit 0
