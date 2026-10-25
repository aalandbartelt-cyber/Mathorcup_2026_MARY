import unittest
from io import BytesIO
from zipfile import ZipFile
from audit_paper_language import scan_blob

def records():return {'files':[],'hits':[],'images':[],'numeric_archives':[],'unhandled':[]}
def archive(mapping):
    b=BytesIO()
    with ZipFile(b,'w') as z:
        for n,data in mapping.items():z.writestr(n,data)
    return b.getvalue()

class PublicationScannerTests(unittest.TestCase):
    def test_split_word_runs_are_joined(self):
        d=archive({'word/document.xml':'<document><p><r><t>E0</t></r><r><t>09</t></r></p></document>'})
        r=records();scan_blob('正文.docx',d,r);self.assertTrue(any(x['term']=='E009' for x in r['hits']))
    def test_nested_zip_contains_docx_and_filename(self):
        d=archive({'图片/agent2.csv':'数值,1','正文.docx':archive({'word/header1.xml':'<p>晋级</p>'})})
        r=records();scan_blob('交付.zip',d,r);self.assertTrue(any(x['term']=='晋级' for x in r['hits']));self.assertTrue(any(x['term']=='agent2' for x in r['hits']))
    def test_academic_words_are_allowed(self):
        r=records();scan_blob('正文.md','基线模型、对比模型、阈值、消融实验、最终模型、稳健性检验'.encode(),r);self.assertEqual(r['hits'],[])
    def test_case_insensitive_management_and_identifiers(self):
        r=records();scan_blob('表.csv',b'x_e003,Promotion,PASS',r);self.assertEqual(len(r['hits']),3)
    def test_metadata_and_relationship_targets_are_scanned(self):
        d=archive({'word/document.xml.rels':'<Relationships><Relationship Target="agent3.png"/></Relationships>','docProps/core.xml':'<title>BZD</title>'})
        r=records();scan_blob('正文.docx',d,r);self.assertEqual(len(r['hits']),2)

if __name__=='__main__':unittest.main()
