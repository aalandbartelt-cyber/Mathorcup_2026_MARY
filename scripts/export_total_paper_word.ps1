param([string]$Source,[string]$OutputDir,[switch]$UpdateToc)
$ErrorActionPreference='Stop'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
$doc=$null
try {
    $doc=$word.Documents.Open($Source,$false,$true)
    $doc.Repaginate()
    if ($UpdateToc) {
        $doc.Fields.Update() | Out-Null
        $toc=$doc.TablesOfContents.Item(1)
        $toc.Update()
        foreach ($p in $toc.Range.Paragraphs) {
            $p.Range.Style=$doc.Styles.Item(-1)
            $p.Range.Font.Name='Times New Roman'
            $p.Range.Font.NameFarEast='宋体'
            $p.Range.Font.Size=10.5
            $p.Range.Font.Bold=0
            $p.Format.FirstLineIndent=0
            $p.Format.LeftIndent=0
            $p.Format.RightIndent=0
            $p.Format.SpaceBefore=0
            $p.Format.SpaceAfter=0
            $p.Format.LineSpacingRule=0
            $p.Format.KeepWithNext=0
            $p.Format.KeepTogether=-1
            $p.Format.TabStops.ClearAll()
            $p.Format.TabStops.Add(436.5,2,1) | Out-Null
        }
        $doc.Repaginate()
        $toc.UpdatePageNumbers()
        $doc.Repaginate()
        [IO.File]::WriteAllText((Join-Path $OutputDir 'toc_fragment.xml'),$toc.Range.WordOpenXML,[Text.UTF8Encoding]::new($false))
    }
    $tables=@()
    for($i=1;$i -le $doc.Tables.Count;$i++) {
        $t=$doc.Tables.Item($i);$s=$t.Range.Duplicate;$s.Collapse(1);$e=$t.Range.Duplicate;$e.End--;$e.Collapse(0)
        $tables += [PSCustomObject]@{index=$i;start_page=$s.Information(3);end_page=$e.Information(3);text=$t.Range.Text.Substring(0,[Math]::Min(100,$t.Range.Text.Length))}
    }
    $heads=@();$caps=@();$tocText=''
    foreach($p in $doc.Paragraphs) {
        $t=$p.Range.Text.Trim()
        if($p.OutlineLevel -le 3) {
            $heads += [PSCustomObject]@{text=$t;number=$p.Range.ListFormat.ListString;physical_page=$p.Range.Information(3);body_page=$p.Range.Information(1);xml=$p.Range.WordOpenXML}
        }
        if($t -match '^[图表]\d+\s') {$caps += [PSCustomObject]@{text=$t;page=$p.Range.Information(3)}}
    }
    $images=@()
    for($i=1;$i -le $doc.InlineShapes.Count;$i++) {$images += [PSCustomObject]@{index=$i;page=$doc.InlineShapes.Item($i).Range.Information(3)}}
    if($doc.TablesOfContents.Count -gt 0){$tocText=$doc.TablesOfContents.Item(1).Range.Text}
    $record=[PSCustomObject]@{source=$Source;pages=$doc.ComputeStatistics(2);formulas=$doc.OMaths.Count;headings=$heads;tables=$tables;captions=$caps;images=$images;toc=$tocText;sections=$doc.Sections.Count}
    $record|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $OutputDir 'word_pagination.json') -Encoding UTF8
    $doc.ExportAsFixedFormat([IO.Path]::ChangeExtension($Source,'.pdf'),17,$false,0)
    Write-Output ('Pages: '+$record.pages+'; equations: '+$record.formulas+'; sections: '+$record.sections)
} finally {
    if($null -ne $doc){$doc.Close($false)}
    $word.Quit()
}
