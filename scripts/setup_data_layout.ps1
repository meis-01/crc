[CmdletBinding()]
param(
    [string]$ExternalRoot = "D:\CODE-II\data",
    [string]$RepositoryDataPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "data")
)

$ErrorActionPreference = "Stop"
$external = [System.IO.Path]::GetFullPath($ExternalRoot)
$repoData = [System.IO.Path]::GetFullPath($RepositoryDataPath)

if (-not $external.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ExternalRoot must be an explicit path on D:, got: $external"
}

foreach ($name in @("raw", "interim", "processed")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $external $name) | Out-Null
}

if (Test-Path -LiteralPath $repoData) {
    $item = Get-Item -LiteralPath $repoData -Force
    if ($item.LinkType -ne "Junction" -or $item.Target -notcontains $external) {
        throw "$repoData already exists and is not the expected junction to $external; refusing to replace it."
    }
} else {
    New-Item -ItemType Junction -Path $repoData -Target $external | Out-Null
}

$drive = Get-PSDrive -Name D
[pscustomobject]@{
    RepositoryDataPath = $repoData
    ExternalDataRoot = $external
    FreeGiB = [math]::Round($drive.Free / 1GB, 2)
} | Format-List

