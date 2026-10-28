using System.Text.Json;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
using DocumentFormat.OpenXml.Wordprocessing;

if (args.Length != 5) throw new ArgumentException("source template spec output audit");
using var spec = JsonDocument.Parse(File.ReadAllText(args[2]));
if (File.Exists(args[3])) throw new IOException("Preserve existing manuscript versions.");
File.Copy(args[0], args[3]);
int addedLines = 0;
using (var original = WordprocessingDocument.Open(args[0], false))
using (var reference = WordprocessingDocument.Open(args[1], false))
using (var document = WordprocessingDocument.Open(args[3], true))
{
    var templateBody = reference.MainDocumentPart!.Document.Body!;
    var sourceElements = templateBody.ChildElements.ToArray();
    var headingTemplate = (Paragraph)sourceElements[774];
    var proseTemplate = (Paragraph)sourceElements[775];
    var subsectionTemplate = (Paragraph)sourceElements[776];
    var codeTemplate = (Paragraph)sourceElements[777];
    if (!headingTemplate.InnerText.StartsWith("附录E") ||
        codeTemplate.ParagraphProperties?.ParagraphStyleId?.Val != "49")
        throw new InvalidDataException("0043 appendix structure has changed.");
    var body = document.MainDocumentPart!.Document.Body!;

    Paragraph Plain(Paragraph prototype, string content)
    {
        var p = (Paragraph)prototype.CloneNode(true);
        foreach (var child in p.ChildElements.Where(x => x is not ParagraphProperties).ToArray()) child.Remove();
        p.Append(new Run(new Text(content) { Space = SpaceProcessingModeValues.Preserve }));
        foreach (var element in p.Descendants().Prepend(p))
            foreach (var a in element.GetAttributes().Where(a => a.LocalName is "paraId" or "textId").ToArray())
                element.RemoveAttribute(a.LocalName, a.NamespaceUri);
        return p;
    }

    void InsertProse(OpenXmlElement anchor, string content)
    {
        var p = Plain(proseTemplate, content);
        var pp = p.ParagraphProperties!;
        pp.ParagraphStyleId = new ParagraphStyleId { Val = "11" };
        // Explicit body properties avoid inheriting the adjacent code style.
        pp.SpacingBetweenLines = new SpacingBetweenLines { Before = "0", After = "60", Line = "300", LineRule = LineSpacingRuleValues.AtLeast };
        pp.Indentation = new Indentation { Left = "0", Right = "0", FirstLine = "480" };
        pp.KeepNext = new KeepNext();
        pp.KeepLines = new KeepLines();
        pp.OutlineLevel = new OutlineLevel { Val = 9 };
        var run = p.Elements<Run>().Single();
        run.PrependChild(new RunProperties(new RunFonts { Ascii = "Times New Roman", HighAnsi = "Times New Roman", EastAsia = "宋体" },
            new FontSize { Val = "21" }, new FontSizeComplexScript { Val = "21" }));
        body.InsertBefore(p, anchor);
    }

    foreach (var section in spec.RootElement.GetProperty("sections").EnumerateArray())
    {
        var name = section.GetProperty("heading").GetString()!;
        var oldHeading = body.Elements<Paragraph>().Single(p => p.InnerText == name);
        if (spec.RootElement.TryGetProperty("replace_existing", out var replace) && replace.GetBoolean())
        {
            var headingNames = spec.RootElement.GetProperty("sections").EnumerateArray()
                .Select(s => s.GetProperty("heading").GetString()).ToHashSet();
            var cursor = oldHeading.NextSibling();
            while (cursor is not null && cursor is not SectionProperties && !headingNames.Contains(cursor.InnerText))
            {
                var next = cursor.NextSibling();
                cursor.Remove();
                cursor = next;
            }
        }
        var heading = Plain(headingTemplate, name);
        var hp = heading.ParagraphProperties!;
        hp.Indentation = new Indentation { Left = "0", Right = "0", FirstLine = "0", FirstLineChars = 0 };
        hp.PageBreakBefore = new PageBreakBefore();
        hp.KeepNext = new KeepNext();
        hp.KeepLines = new KeepLines();
        hp.OutlineLevel = new OutlineLevel { Val = 9 };
        body.InsertBefore(heading, oldHeading);
        InsertProse(oldHeading, section.GetProperty("intro").GetString()!);

        foreach (var block in section.GetProperty("blocks").EnumerateArray())
        {
            var subheading = Plain(subsectionTemplate, block.GetProperty("title").GetString()!);
            var sp = subheading.ParagraphProperties!;
            sp.KeepNext = new KeepNext();
            sp.KeepLines = new KeepLines();
            sp.OutlineLevel = new OutlineLevel { Val = 9 };
            sp.Indentation = new Indentation { Left = "0", Right = "0", FirstLine = "0", FirstLineChars = 0 };
            body.InsertBefore(subheading, oldHeading);
            InsertProse(oldHeading, block.GetProperty("software").GetString()!);
            foreach (var context in block.GetProperty("context").EnumerateArray())
                InsertProse(oldHeading, context.GetString()!);
            var lines = block.GetProperty("lines").EnumerateArray().ToArray();
            for (int index = 0; index < lines.Length; index++)
            {
                var line = lines[index];
                var p = Plain(codeTemplate, "");
                p.RemoveAllChildren<Run>();
                var pp = p.ParagraphProperties!;
                pp.ParagraphStyleId = new ParagraphStyleId { Val = "49" };
                pp.KeepNext = new KeepNext { Val = false };
                pp.KeepLines = new KeepLines();
                pp.OutlineLevel = new OutlineLevel { Val = 9 };
                pp.WordWrap = new WordWrap { Val = true };
                pp.SnapToGrid = new SnapToGrid { Val = false };
                pp.Indentation = new Indentation { Left = "0", Right = "0", FirstLine = "0", FirstLineChars = 0 };
                pp.Justification = new Justification { Val = JustificationValues.Left };
                pp.SpacingBetweenLines = new SpacingBetweenLines { Before = "0", After = "0", Line = "300", LineRule = LineSpacingRuleValues.AtLeast };
                foreach (var span in line.GetProperty("spans").EnumerateArray())
                {
                    bool key = span.GetProperty("keyword").GetBoolean();
                    var rp = new RunProperties(
                        new RunFonts { Ascii = "Consolas", HighAnsi = "Consolas", EastAsia = "宋体", ComplexScript = "宋体" },
                        new Bold { Val = key }, new BoldComplexScript { Val = key },
                        new NoProof(), new Color { Val = span.TryGetProperty("color", out var ink) ? ink.GetString()! : key ? "006699" : "333333" },
                        new FontSize { Val = "20" }, new FontSizeComplexScript { Val = "20" });
                    p.Append(new Run(rp, new Text(span.GetProperty("text").GetString()!) { Space = SpaceProcessingModeValues.Preserve }));
                }
                body.InsertBefore(p, oldHeading);
                addedLines++;
            }
        }
        oldHeading.Remove();
    }
    document.MainDocumentPart.Document.Save();
    var validator = new OpenXmlValidator(FileFormatVersions.Microsoft365);
    var oldErrors = validator.Validate(original).Select(e => new { e.Id, e.Description, Part = e.Part?.Uri.ToString(), Path = e.Path?.XPath }).ToArray();
    var errors = validator.Validate(document).Select(e => new { e.Id, e.Description, Part = e.Part?.Uri.ToString(), Path = e.Path?.XPath }).ToArray();
    var oldCounts = oldErrors.GroupBy(e => (e.Id, e.Description, e.Part)).ToDictionary(g => g.Key, g => g.Count());
    int extra = errors.GroupBy(e => (e.Id, e.Description, e.Part)).Sum(g => Math.Max(0, g.Count() - oldCounts.GetValueOrDefault(g.Key)));
    File.WriteAllText(args[4], JsonSerializer.Serialize(new { errors, originalErrorCount = oldErrors.Length, errorCount = errors.Length, newErrorCount = extra, addedLines }, new JsonSerializerOptions { WriteIndented = true }));
    if (extra > 0) throw new InvalidDataException($"New SDK errors: {extra}. See audit.");
    Console.WriteLine($"Appended {addedLines} source lines, original SDK findings {oldErrors.Length}, output {errors.Length}, new {extra}.");
}
