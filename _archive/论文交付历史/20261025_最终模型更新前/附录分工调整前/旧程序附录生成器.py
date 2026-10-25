"""Create a native Word appendix containing the standalone program sources."""
from pathlib import Path
from docx.shared import Pt
from docx.oxml.ns import qn
from build_q1_final_publication import base_document,save,OUT,add_heading,add_body

def main():
    doc,_=base_document('附录 第一问源程序')
    add_heading(doc,'附录 第一问源程序',1)
    add_body(doc,'运行环境为Python 3.10和R。Python依赖numpy、pandas、scipy、scikit-learn；R依赖ggplot2、ragg、dplyr、tidyr、readr、patchwork、svglite。全部模型程序、固定划分和绘图数据随支撑材料提供，具体命令见程序目录README。以下按文件名列出源程序；Word中的自动换行不改变支撑材料文件。')
    files=sorted(p for p in (OUT/'程序').iterdir() if p.suffix in {'.py','.R','.json','.txt'})
    for path in files:
        add_heading(doc,path.name,2)
        for line in path.read_text().splitlines():
            p=doc.add_paragraph(style='Body Text')
            p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.line_spacing=Pt(10.5);p.paragraph_format.keep_together=False
            p.paragraph_format.widow_control=True
            run=p.add_run(line if line else ' ');run.font.name='Consolas';run.font.size=Pt(8)
            run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    output=save(doc,OUT/'附录_第一问源程序_v2.docx');print(output)

if __name__=='__main__':
    raise SystemExit('按用户最新分工，不再生成程序附录文档；打包时提供附录核心代码，由论文手排版。')
