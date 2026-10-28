param([string]$Source,[string]$OutputPdf,[string]$AuditJson)
$ErrorActionPreference='Stop'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
$doc=$null
try {
    $doc=$word.Documents.Open($Source,$false,$true)
    $doc.Repaginate()
    $record=[PSCustomObject]@{
        source=$Source
        pages=$doc.ComputeStatistics(2)
        formulas=$doc.OMaths.Count
        sections=$doc.Sections.Count
        tables=$doc.Tables.Count
        images=$doc.InlineShapes.Count
        toc_count=$doc.TablesOfContents.Count
    }
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $AuditJson -Encoding UTF8
    $doc.ExportAsFixedFormat($OutputPdf,17,$false,0)
    Write-Output ('Pages: '+$record.pages+'; equations: '+$record.formulas+'; sections: '+$record.sections+'; toc: '+$record.toc_count)
} finally {
    if($null -ne $doc){$doc.Close($false)}
    $word.Quit()
}
