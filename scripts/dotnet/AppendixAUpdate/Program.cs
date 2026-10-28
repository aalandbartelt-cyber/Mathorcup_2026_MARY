using System.Text.Json;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
using DocumentFormat.OpenXml.Wordprocessing;

if (args.Length != 5)
    throw new ArgumentException("input.docx template.docx spec.json output.docx audit.json");
string sourcePath = args[0], templatePath = args[1], specPath = args[2], outputPath = args[3], auditPath = args[4];
using var specFile = JsonDocument.Parse(File.ReadAllText(specPath));
var spec = specFile.RootElement;
var overview = spec.GetProperty("overview").EnumerateArray()
    .Select(r => r.EnumerateArray().Select(c => c.GetString() ?? "").ToArray()).ToArray();
var programs = spec.GetProperty("programs").EnumerateArray()
    .Select(r => r.EnumerateArray().Select(c => c.GetString() ?? "").ToArray()).ToArray();
if (File.Exists(outputPath)) throw new IOException("Refusing to overwrite an existing manuscript.");
File.Copy(sourcePath, outputPath, false);

using var template = WordprocessingDocument.Open(templatePath, false);
using var output = WordprocessingDocument.Open(outputPath, true);
using var original = WordprocessingDocument.Open(sourcePath, false);
var templateBody = template.MainDocumentPart!.Document.Body!;
var body = output.MainDocumentPart!.Document.Body!;
var placeholder = body.Elements<Paragraph>().Single(p => p.InnerText == "待支撑材料整理后补充");
var heading = placeholder.PreviousSibling<Paragraph>()!;
if (heading.InnerText != "附录A 支撑材料文件列表") throw new InvalidDataException("Wrong appendix anchor.");
var nextHeading = placeholder.NextSibling<Paragraph>()!;
if (nextHeading.InnerText != "附录B 补充表格与图片") throw new InvalidDataException("Wrong appendix end anchor.");

Paragraph TemplateCaption(string text) => templateBody.Elements<Paragraph>().Single(p => p.InnerText == text);
Table TableAfter(Paragraph paragraph) => paragraph.ElementsAfter().OfType<Table>().First();
var caption1Template = TemplateCaption("表A.1 支撑材料文件列表");
var caption2Template = TemplateCaption("表A.2 程序文件与支撑材料对照表");
var table1Template = TableAfter(caption1Template);
var table2Template = TableAfter(caption2Template);

Paragraph WithText(Paragraph prototype, string text)
{
    var clone = (Paragraph)prototype.CloneNode(true);
    var originalRun = clone.Elements<Run>().FirstOrDefault();
    var runProps = originalRun?.RunProperties is null ? null : (RunProperties)originalRun.RunProperties.CloneNode(true);
    foreach (var child in clone.ChildElements.Where(c => c is not ParagraphProperties).ToArray()) child.Remove();
    var run = new Run();
    if (runProps is not null) run.Append(runProps);
    run.Append(new Text(text) { Space = SpaceProcessingModeValues.Preserve });
    clone.Append(run);
    return clone;
}

Table MakeTable(Table prototype, string[][] content, int[] widths, bool keepWhole)
{
    var clone = (Table)prototype.CloneNode(true);
    var headerPrototype = prototype.Elements<TableRow>().First();
    var dataPrototype = prototype.Elements<TableRow>().Skip(1).First();
    clone.RemoveAllChildren<TableRow>();
    var props = clone.GetFirstChild<TableProperties>()!;
    props.TableWidth = new TableWidth { Width = widths.Sum().ToString(), Type = TableWidthUnitValues.Dxa };
    props.TableLayout = new TableLayout { Type = TableLayoutValues.Fixed };
    var grid = clone.GetFirstChild<TableGrid>()!;
    grid.RemoveAllChildren();
    foreach (var width in widths) grid.Append(new GridColumn { Width = width.ToString() });
    for (int r = 0; r < content.Length; r++)
    {
        var row = (TableRow)(r == 0 ? headerPrototype : dataPrototype).CloneNode(true);
        var rowProps = row.GetFirstChild<TableRowProperties>();
        if (rowProps is null) { rowProps = new TableRowProperties(); row.PrependChild(rowProps); }
        rowProps.RemoveAllChildren<TableRowHeight>();
        rowProps.RemoveAllChildren<TableHeader>();
        rowProps.RemoveAllChildren<CantSplit>();
        rowProps.AddChild(new CantSplit(), true);
        if (r == 0) rowProps.AddChild(new TableHeader(), true);
        var cells = row.Elements<TableCell>().ToArray();
        if (cells.Length != widths.Length || content[r].Length != widths.Length)
            throw new InvalidDataException("Table column count mismatch.");
        for (int c = 0; c < widths.Length; c++)
        {
            var cell = cells[c];
            var cp = cell.GetFirstChild<TableCellProperties>();
            if (cp is null) { cp = new TableCellProperties(); cell.PrependChild(cp); }
            cp.TableCellWidth = new TableCellWidth { Width = widths[c].ToString(), Type = TableWidthUnitValues.Dxa };
            cp.TableCellVerticalAlignment = new TableCellVerticalAlignment { Val = TableVerticalAlignmentValues.Center };
            var p = WithText(cell.Elements<Paragraph>().First(), content[r][c]);
            cell.RemoveAllChildren<Paragraph>();
            var pp = p.ParagraphProperties ?? p.PrependChild(new ParagraphProperties());
            pp.SpacingBetweenLines = new SpacingBetweenLines { Before = "0", After = "0", Line = "240", LineRule = LineSpacingRuleValues.Auto };
            pp.Indentation = new Indentation { Left = "0", Right = "0", FirstLine = "0" };
            pp.Justification = new Justification { Val = c == 0 || (widths.Length == 3 && c == 2) ? JustificationValues.Center : JustificationValues.Left };
            pp.KeepLines = new KeepLines();
            pp.KeepNext = new KeepNext { Val = keepWhole && r < content.Length - 1 };
            foreach (var run in p.Elements<Run>())
            {
                var rp = run.RunProperties ?? run.PrependChild(new RunProperties());
                rp.Bold = new Bold { Val = r == 0 };
                rp.BoldComplexScript = new BoldComplexScript { Val = r == 0 };
                rp.RunFonts = new RunFonts { Ascii = "Times New Roman", HighAnsi = "Times New Roman", EastAsia = "宋体" };
                rp.FontSize = new FontSize { Val = "21" };
                rp.FontSizeComplexScript = new FontSizeComplexScript { Val = "21" };
            }
            cell.Append(p);
        }
        clone.Append(row);
    }
    return clone;
}

var intro = WithText(placeholder, spec.GetProperty("intro").GetString()!);
var caption1 = WithText(caption1Template, "表A.1 支撑材料文件列表");
var caption2 = WithText(caption2Template, "表A.2 程序文件与支撑材料对照表");
caption1.ParagraphProperties ??= new ParagraphProperties();
caption2.ParagraphProperties ??= new ParagraphProperties();
caption1.ParagraphProperties.KeepNext = new KeepNext();
caption2.ParagraphProperties.KeepNext = new KeepNext();
// Each list fits one page; keep the source inventory together on its own page.
caption2.ParagraphProperties.PageBreakBefore = new PageBreakBefore();
var table1 = MakeTable(table1Template, overview, [700, 3500, 1320, 3120], true);
var table2 = MakeTable(table2Template, programs, [700, 6640, 1300], true);

var insert = new OpenXmlElement[] { intro, caption1, table1, caption2, table2 };
foreach (var element in insert.SelectMany(e => e.Descendants().Prepend(e)))
{
    foreach (var attribute in element.GetAttributes().Where(a => a.LocalName is "paraId" or "textId").ToArray())
        element.RemoveAttribute(attribute.LocalName, attribute.NamespaceUri);
}
foreach (var element in insert) body.InsertBefore(element, placeholder);
placeholder.Remove();
output.MainDocumentPart.Document.Save();

var validator = new OpenXmlValidator(FileFormatVersions.Microsoft365);
var originalErrors = validator.Validate(original).Select(e => new {
    e.Id, e.Description, Path = e.Path?.XPath, Part = e.Part?.Uri.ToString()
}).ToArray();
var errors = validator.Validate(output).Select(e => new {
    e.Id, e.Description, Path = e.Path?.XPath, Part = e.Part?.Uri.ToString()
}).ToArray();
var originalCounts = originalErrors.GroupBy(e => (e.Id, e.Description, e.Part))
    .ToDictionary(g => g.Key, g => g.Count());
var extraErrorCount = errors.GroupBy(e => (e.Id, e.Description, e.Part))
    .Sum(g => Math.Max(0, g.Count() - originalCounts.GetValueOrDefault(g.Key)));
File.WriteAllText(auditPath, JsonSerializer.Serialize(new {
    tool = "OpenXML SDK 3.5.1", errors, errorCount = errors.Length,
    originalErrors, originalErrorCount = originalErrors.Length,
    newErrorCount = extraErrorCount,
    appendixOverviewRows = overview.Length - 1,
    appendixProgramRows = programs.Length - 1,
    templateStyles = new[] { "41", "45" },
    tableWidths = new[] { 8640, 8640 },
    copiedTemplateTables = true,
    output = Path.GetFileName(outputPath)
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Created {outputPath}; A.1={overview.Length - 1}, A.2={programs.Length - 1}, original SDK findings={originalErrors.Length}, output findings={errors.Length}, new findings={extraErrorCount}");
if (extraErrorCount > 0) Environment.Exit(2);
