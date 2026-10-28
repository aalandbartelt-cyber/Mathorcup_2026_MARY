using System.Text.Json;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
using DocumentFormat.OpenXml.Wordprocessing;

if (args.Length != 4) throw new ArgumentException("source.docx spec.json output.docx audit.json");
using var specFile = JsonDocument.Parse(File.ReadAllText(args[1]));
var spec = specFile.RootElement;
if (File.Exists(args[2])) throw new IOException("Do not overwrite existing manuscript editions.");
File.Copy(args[0], args[2]);
using var original = WordprocessingDocument.Open(args[0], false);
using (var output = WordprocessingDocument.Open(args[2], true))
{
    var body = output.MainDocumentPart!.Document.Body!;
    var styles = output.MainDocumentPart.StyleDefinitionsPart!.Styles!;
    void NonTocStyle(string prototypeId, string id, string name)
    {
        if (styles.Elements<Style>().Any(s => s.StyleId == id)) return;
        var copy = (Style)styles.Elements<Style>().Single(s => s.StyleId == prototypeId).CloneNode(true);
        copy.StyleId = id;copy.StyleName = new StyleName { Val = name };
        copy.CustomStyle = true;
        copy.RemoveAllChildren<LinkedStyle>();
        copy.RemoveAllChildren<UIPriority>();
        copy.AddChild(new UIPriority { Val = 9 }, true);
        copy.StyleParagraphProperties ??= new StyleParagraphProperties();
        copy.StyleParagraphProperties.OutlineLevel = new OutlineLevel { Val = 9 };
        styles.Append(copy);
    }
    NonTocStyle("3", "AppendixCodeSubheading", "Appendix code subheading");
    NonTocStyle("2", "AppendixUmbrella", "Appendix umbrella title");
    var paragraphs = body.Elements<Paragraph>().ToArray();
    var codeTemplate = paragraphs.First(p => p.ParagraphProperties?.ParagraphStyleId?.Val == "49");
    var subTemplate = paragraphs.First(p => p.InnerText.StartsWith("C.1 "));
    var proseTemplate = paragraphs.First(p => p.InnerText.StartsWith("本附录给出感官组"));
    var mainBefore = paragraphs.First(p => p.InnerText == "附录C 问题一模型核心程序");
    var referenceHeading = paragraphs.Single(p => p.InnerText == "参考文献");
    var refs = referenceHeading.ElementsAfter().OfType<Paragraph>().TakeWhile(p => p.InnerText != "AI 工具使用声明").Where(p => Regex.IsMatch(p.InnerText, @"^\[\d+\]")).ToArray();
    if (refs.Length != 14) throw new InvalidDataException("Expected fourteen references.");

    string TextOf(Paragraph p) => string.Concat(p.Descendants<Text>().Select(t => t.Text));
    void Replace(Paragraph p, string find, string value)
    {
        var nodes = p.Descendants<Text>().ToArray();
        var text = string.Concat(nodes.Select(t => t.Text));
        var offsets = new int[nodes.Length];
        for (int i = 1; i < nodes.Length; i++) offsets[i] = offsets[i - 1] + nodes[i - 1].Text.Length;
        int from = text.Length;
        while (from > 0)
        {
            int pos = text.LastIndexOf(find, from - 1, StringComparison.Ordinal);
            if (pos < 0) break;
            int start = Array.FindLastIndex(offsets, x => x <= pos);
            int end = Array.FindLastIndex(offsets, x => x < pos + find.Length);
            string lead = nodes[start].Text[..(pos - offsets[start])];
            string tail = nodes[end].Text[(pos + find.Length - offsets[end])..];
            nodes[start].Text = lead + value + tail;
            nodes[start].Space = SpaceProcessingModeValues.Preserve;
            for (int i = start + 1; i <= end; i++) nodes[i].Text = "";
            from = pos;
        }
    }
    Paragraph Plain(Paragraph proto, string text)
    {
        var p = (Paragraph)proto.CloneNode(true);
        var rp = p.Elements<Run>().FirstOrDefault()?.RunProperties?.CloneNode(true) as RunProperties;
        foreach (var c in p.ChildElements.Where(c => c is not ParagraphProperties).ToArray()) c.Remove();
        var run = new Run();
        if (rp is not null) run.Append(rp);
        run.Append(new Text(text) { Space = SpaceProcessingModeValues.Preserve });
        p.Append(run);
        foreach (var e in p.Descendants().Prepend(p))
            foreach (var a in e.GetAttributes().Where(a => a.LocalName is "paraId" or "textId").ToArray())
                e.RemoveAttribute(a.LocalName, a.NamespaceUri);
        return p;
    }

    // Insert exact, locally defined parameters before wording/citation rewrites.
    foreach (var item in spec.GetProperty("insertions").EnumerateArray())
    {
        var prefix = item.GetProperty("prefix").GetString()!;
        var anchor = body.Elements<Paragraph>().Single(p => p.InnerText.StartsWith(prefix));
        var inserted = new Paragraph(item.GetProperty("xml").GetString()!);
        if (item.GetProperty("position").GetString() == "before") body.InsertBefore(inserted, anchor);
        else body.InsertAfter(inserted, anchor);
    }
    foreach (var item in spec.GetProperty("paragraph_replacements").EnumerateArray())
    {
        var anchor = body.Elements<Paragraph>().Single(p => p.InnerText.StartsWith(item.GetProperty("prefix").GetString()!));
        body.InsertBefore(new Paragraph(item.GetProperty("xml").GetString()!), anchor);
        anchor.Remove();
    }
    bool beforeReferences = true;
    foreach (var block in body.ChildElements.ToArray())
    {
        if (block == referenceHeading) beforeReferences = false;
        if (!beforeReferences) break;
        foreach (var p in block.Descendants<Paragraph>().Prepend(block as Paragraph).OfType<Paragraph>().Distinct())
        {
            var matches = Regex.Matches(TextOf(p), @"\[(\d+)\]").ToArray();
            // Placeholders prevent new numbers from being substituted twice.
            foreach (var match in matches.Reverse())
                Replace(p, match.Value, "{REF:" + match.Groups[1].Value + "}");
            foreach (var pair in spec.GetProperty("reference_map").EnumerateObject())
                Replace(p, "{REF:" + pair.Name + "}", "[" + pair.Value.GetInt32() + "]");
        }
    }
    foreach (var item in spec.GetProperty("replacements").EnumerateArray())
        foreach (var p in body.Descendants<Paragraph>().ToArray())
            Replace(p, item.GetProperty("find").GetString()!, item.GetProperty("replace").GetString()!);
    var words = spec.GetProperty("global_words").EnumerateObject().ToArray();
    foreach (var block in body.ChildElements.TakeWhile(b => b != mainBefore).ToArray())
        foreach (var p in block.Descendants<Paragraph>().Prepend(block as Paragraph).OfType<Paragraph>().Distinct())
            foreach (var pair in words)
            {
                if (TextOf(p).Contains(".py") || TextOf(p).Contains(".R")) continue;
                Replace(p, pair.Name, pair.Value.GetString()!);
            }
    foreach (var (entry, index) in spec.GetProperty("references").EnumerateArray().OrderBy(x => x.GetProperty("new").GetInt32()).Select((x, i) => (x, i)))
    {
        var replacement = Plain(refs[index], entry.GetProperty("citation").GetString()!);
        body.InsertBefore(replacement, refs[index]);
        refs[index].Remove();
    }
    // Table 14: only explanatory rules are added; no result cell is changed.
    var caption14 = body.Elements<Paragraph>().Single(p => p.InnerText.StartsWith("表14 "));
    var table14 = caption14.ElementsAfter().OfType<Table>().First();
    var rowProto = table14.Elements<TableRow>().Last();
    foreach (var row in spec.GetProperty("table14_rows").EnumerateArray())
    {
        var tr = (TableRow)rowProto.CloneNode(true);
        var values = row.EnumerateArray().Select(x => x.GetString()!).ToArray();
        foreach (var (cell, index) in tr.Elements<TableCell>().Select((x, i) => (x, i)))
        {
            var p = Plain(cell.Elements<Paragraph>().First(), values[index]);
            cell.RemoveAllChildren<Paragraph>();
            cell.Append(p);
        }
        table14.Append(tr);
    }
    // Appendix A must not claim that the absent AI PDF was packaged.
    var aiCell = body.Descendants<TableCell>().FirstOrDefault(c => c.InnerText == "AI工具使用详情说明");
    if (aiCell is not null)
    {
        var cells = aiCell.Parent!.Elements<TableCell>().ToArray();
        var p = Plain(cells[^1].Elements<Paragraph>().First(), "AI工具使用详情另行补充，未计入现有附件包");
        cells[^1].RemoveAllChildren<Paragraph>();cells[^1].Append(p);
    }

    // Replace every previous appendix block using exact shipped source excerpts.
    var names = spec.GetProperty("sections").EnumerateArray().Select(s => s.GetProperty("heading").GetString()).ToHashSet();
    foreach (var section in spec.GetProperty("sections").EnumerateArray())
    {
        var oldHeading = body.Elements<Paragraph>().Single(p => p.InnerText == section.GetProperty("heading").GetString());
        var cursor = oldHeading.NextSibling();
        while (cursor is not null && cursor is not SectionProperties && !names.Contains(cursor.InnerText))
        {
            var next = cursor.NextSibling();cursor.Remove();cursor = next;
        }
        var pp = oldHeading.ParagraphProperties!;
        pp.PageBreakBefore = new PageBreakBefore();pp.OutlineLevel = new OutlineLevel { Val = 1 };
        var intro = Plain(proseTemplate, section.GetProperty("intro").GetString()!);
        body.InsertAfter(intro, oldHeading);
        OpenXmlElement anchor = intro;
        foreach (var block in section.GetProperty("blocks").EnumerateArray())
        {
            var h = Plain(subTemplate, block.GetProperty("title").GetString()!);
            h.ParagraphProperties!.ParagraphStyleId = new ParagraphStyleId { Val = "AppendixCodeSubheading" };
            h.ParagraphProperties!.OutlineLevel = new OutlineLevel { Val = 9 };
            body.InsertAfter(h, anchor);anchor = h;
            var software = Plain(proseTemplate, block.GetProperty("software").GetString()!);
            body.InsertAfter(software, anchor);anchor = software;
            foreach (var context in block.GetProperty("context").EnumerateArray())
            {
                var p = Plain(proseTemplate, context.GetString()!);body.InsertAfter(p, anchor);anchor = p;
            }
            foreach (var line in block.GetProperty("lines").EnumerateArray())
            {
                var p = Plain(codeTemplate, "");p.RemoveAllChildren<Run>();
                foreach (var span in line.GetProperty("spans").EnumerateArray())
                {
                    bool bold = span.GetProperty("keyword").GetBoolean();
                    var rp = new RunProperties(
                        new RunFonts { Ascii = "Consolas", HighAnsi = "Consolas", EastAsia = "宋体" },
                        new Bold { Val = bold }, new BoldComplexScript { Val = bold },new NoProof(),
                        new Color { Val = span.GetProperty("color").GetString()! },
                        new FontSize { Val = "20" }, new FontSizeComplexScript { Val = "20" });
                    p.Append(new Run(rp, new Text(span.GetProperty("text").GetString()!) { Space = SpaceProcessingModeValues.Preserve }));
                }
                body.InsertAfter(p, anchor);anchor = p;
            }
        }
    }

    // Install a genuine Word TOC; include only A-E, not code subsections or the AI notice.
    var contentsTitle = body.Elements<Paragraph>().Single(p => p.InnerText == "目录");
    var stop = contentsTitle.ElementsAfter().OfType<Paragraph>().First(p => p.ParagraphProperties?.SectionProperties is not null);
    var cache = contentsTitle.NextSibling();
    while (cache is not null && cache != stop) { var next = cache.NextSibling();cache.Remove();cache = next; }
    var toc = new Paragraph(
        new Run(new FieldChar { FieldCharType = FieldCharValues.Begin }),
        new Run(new FieldCode(" TOC \\o \"1-2\" \\h \\z \\u ") { Space = SpaceProcessingModeValues.Preserve }),
        new Run(new FieldChar { FieldCharType = FieldCharValues.Separate }),
        new Run(new Text("目录将在Word中更新")),
        new Run(new FieldChar { FieldCharType = FieldCharValues.End }));
    body.InsertAfter(toc, contentsTitle);
    bool appendix = false;
    foreach (var p in body.Elements<Paragraph>())
    {
        if (p.InnerText == "附录") { appendix = true;p.ParagraphProperties!.ParagraphStyleId = new ParagraphStyleId { Val = "AppendixUmbrella" };p.ParagraphProperties!.OutlineLevel = new OutlineLevel { Val = 9 }; }
        if (!appendix) continue;
        if (Regex.IsMatch(p.InnerText, @"^附录[A-E] "))
            p.ParagraphProperties!.OutlineLevel = new OutlineLevel { Val = 1 };
        else if (p.InnerText.StartsWith("B.1 "))
            p.ParagraphProperties!.OutlineLevel = new OutlineLevel { Val = 9 };
    }
    output.MainDocumentPart.Document.Save();
    styles.Save();
    var validator = new OpenXmlValidator(FileFormatVersions.Microsoft365);
    var oldErrors = validator.Validate(original).Select(e => new { e.Id, e.Description, Part = e.Part?.Uri.ToString() }).ToArray();
    var errors = validator.Validate(output).Select(e => new { e.Id, e.Description, Part = e.Part?.Uri.ToString(), Path = e.Path?.XPath }).ToArray();
    var counts = oldErrors.GroupBy(e => (e.Id, e.Description, e.Part)).ToDictionary(g => g.Key, g => g.Count());
    var added = errors.GroupBy(e => (e.Id, e.Description, e.Part)).Sum(g => Math.Max(0, g.Count() - counts.GetValueOrDefault(g.Key)));
    File.WriteAllText(args[3], JsonSerializer.Serialize(new { originalCount = oldErrors.Length, count = errors.Length, newErrorCount = added, errors }, new JsonSerializerOptions { WriteIndented = true }));
    if (added > 0) throw new InvalidDataException($"New SDK errors {added}; see audit output");
    Console.WriteLine($"V13 written. New SDK findings {added}; inherited {oldErrors.Length}.");
}
