"""Record the user-review checkpoint, without authorizing figure insertion."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import csv
import json
import shutil
from lxml import etree as E
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'05_论文/总论文';AUDIT=ROOT/'docs/collab/20261027_总论文v2准备'
DRAFT=BASE/'v2_准备稿';FIG=BASE/'图稿_待审计/数据概览'


def main():
    checks=json.loads((AUDIT/'图前工作稿验收.json').read_text());assert checks['ready_for_user_figure_review']
    assert json.loads((AUDIT/'SDK工作稿校验.json').read_text(encoding='utf-8-sig'))['isValid']
    doc=DRAFT/'总论文_v2_图前工作稿.docx'
    with ZipFile(doc) as z:files={n:z.read(n) for n in z.namelist()}
    app=E.fromstring(files['docProps/app.xml']);n='{'+E.QName(app).namespace+'}'
    if app.find(n+'Pages').text!=str(checks['physical_pages']):
        app.find(n+'Pages').text=str(checks['physical_pages'])
        files['docProps/app.xml']=E.tostring(app,encoding='UTF-8',xml_declaration=True,standalone=True)
        with ZipFile(doc,'w',ZIP_DEFLATED) as z:
            for name,data in files.items():z.writestr(name,data)
    sources=AUDIT/'附件使用审计'
    for name in ['附件使用矩阵.csv','附件使用矩阵_长表.csv','附件口径汇总.csv']:
        shutil.copy2(sources/name,FIG/name)
    manifests=[]
    panels=[
      '9字段缺失数；50目标家族；22风格真实样本量',
      '231字段候选资格；218单体成对相关；13排除字段类别',
      '三风格真实样本；六维实测中位/IQR；已有画像有限值计数']
    captions=['感官预测数据的缺失结构、目标构成与风格样本量',
              '单体候选资格及其实际共线性分布',
              '目标风格的真实样本、实测感官分布与画像接口完整性']
    for idx in range(1,4):
        png=FIG/f'Q{idx}_数据概览_待审.png';pdf=png.with_suffix('.pdf')
        manifests.append({'draft_id':f'Q{idx}_数据概览','status':'待用户审计_未插正文',
          'caption_draft':captions[idx-1],'panels':panels[idx-1],'png':str(png.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),
          'png_sha256':sha256(png.read_bytes()).hexdigest(),'pdf_sha256':sha256(pdf.read_bytes()).hexdigest()})
    with (FIG/'图清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=manifests[0]);w.writeheader();w.writerows(manifests)
    sums=[]
    for p in sorted(FIG.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':sums.append(sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(FIG)))
    (FIG/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    readme='''# 总论文v2图前工作稿

本目录为图前工作稿，尚不是最终v2；三张数据概览仍在独立图稿目录待用户审计。

- DOCX由v1副本修改，Word原生导出同源PDF；v1及其说明均保持保护哈希。
- 共30物理页：前置2页、正文含参考文献25页、附录3页。题目、摘要、关键词仍为原占位。
- 符号表按用户最新要求由20组减为10组，只留核心符号。全部符号列使用原生OMML；23个独立公式后仍有就地解释，完整81概念台账留在内部审计。
- 参考文献[12][13]位于物理第27页（正文第25页），工具型号按用户提供值保留，未增加其他工具。
- AI声明标题位于同页；候选原文未作为最终声明采用。实际记录包含模型候选、代码与检验辅助，需按真实范围核对；候选原文完整保存在docs/collab的AI披露待核.md。
- 附录A为待补文件列表；B已有S1—S3和逐图阅读说明，Q2/Q3明确引用正文对应检验图表；C/D/E仅保留核心程序标题、不含代码。
- 图前稿全部图表页、目录已核对；19张结果表数字、14幅原主图字节均不变。新数据图尚未插入。
- 该稿在图审计和披露核对完成前不可作为正式提交稿。未触及06_交付，未进行Git提交或任何模型训练。
'''
    refs_page=checks['ai_reference_physical_pages']['[12]'][0]
    readme=readme.replace('共30物理页：前置2页、正文含参考文献25页、附录3页。',
      f'共{checks["physical_pages"]}物理页：前置2页、正文含参考文献{checks["body_pages"]}页，其后为附录。')
    readme=readme.replace('物理第27页（正文第25页）',f'物理第{refs_page}页（正文第{refs_page-2}页）')
    readme+=f'\n当前数学对象总数{checks["omml_count"]}，空run为{checks["empty_math_runs"]}。\n'
    (DRAFT/'图前工作稿说明.md').write_text(readme)
    record=readme+'\n## 当前用户审计入口\n\n'
    for row in manifests:record+=f'- {row["draft_id"]}：{row["png"]}\n'
    record+='\n附件使用审计共有24格矩阵，逐表形状/单位和代码消费证据同目录保留。附件5用于Q1目标元数据；附件2在Q2仅作全量描述/核验；Q3附件1/8直接、附件4经最终OOF接口，正式Q2 Top-3未进入优化。\n'
    record+=f'\n## 验收范围与未决项\n\n{len(checks["gates"])}项图前技术检查通过，SDK错误0/警告0。用户尚未审计三图；AI披露尚未按完整真实记录定稿。这两项均不以自动检查或“继续”代替确认。\n'
    (AUDIT/'图前检查点记录.md').write_text(record)
    filechecks={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()} for p in [doc,doc.with_suffix('.pdf'),DRAFT/'图前工作稿说明.md']}
    (DRAFT/'SHA256SUMS').write_text('\n'.join(v['sha256']+'  '+name for name,v in filechecks.items())+'\n')
    (AUDIT/'图前产物哈希.json').write_text(json.dumps({'files':filechecks,'figures':manifests},ensure_ascii=False,indent=2))
    print('Pre-figure checkpoint recorded; awaiting user review of 3 drafts.')


if __name__=='__main__':main()
