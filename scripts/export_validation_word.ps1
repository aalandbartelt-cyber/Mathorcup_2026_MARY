param([Parameter(Mandatory=$true)][string]$JobFile)
$ErrorActionPreference = 'Stop'
$jobs = Get-Content -LiteralPath $JobFile -Raw -Encoding UTF8 | ConvertFrom-Json
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$records = @()
try {
    foreach ($job in $jobs) {
        $doc = $null
        try {
            $doc = $word.Documents.Open([string]$job.source, $false, $true)
            $doc.Repaginate()
            $pdf = [IO.Path]::ChangeExtension([string]$job.source, '.pdf')
            $doc.ExportAsFixedFormat($pdf, 17, $false, 0)
            $tables = @()
            for ($i=1; $i -le $doc.Tables.Count; $i++) {
                $t = $doc.Tables.Item($i)
                $s = $t.Range.Duplicate; $s.Collapse(1)
                $e = $t.Range.Duplicate; $e.End = $e.End - 1; $e.Collapse(0)
                $tables += [PSCustomObject]@{index=$i; start_page=$s.Information(3); end_page=$e.Information(3); text=$t.Range.Text.Substring(0,[Math]::Min(110,$t.Range.Text.Length))}
            }
            $captions = @()
            foreach ($p in $doc.Paragraphs) {
                $text = $p.Range.Text.Trim()
                if ($text -match '^(附图S\d|[图表]\d)\s') {
                    $captions += [PSCustomObject]@{text=$text; page=$p.Range.Information(3)}
                }
            }
            $images = @()
            for ($i=1; $i -le $doc.InlineShapes.Count; $i++) {
                $images += [PSCustomObject]@{index=$i; page=$doc.InlineShapes.Item($i).Range.Information(3)}
            }
            $records += [PSCustomObject]@{source=$job.source; pdf=$pdf; pages=$doc.ComputeStatistics(2); formulas=$doc.OMaths.Count; tables=$tables; captions=$captions; images=$images}
        } finally {
            if ($null -ne $doc) { $doc.Close($false) }
        }
    }
} finally {
    $word.Quit()
}
$records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ([IO.Path]::Combine([IO.Path]::GetDirectoryName($JobFile),'word_pagination.json')) -Encoding UTF8
Write-Output ('Exported '+$records.Count+' documents.')
