"""Write the integration handoff from checked Word/PDF artifacts."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re
from lxml import etree as E
import pymupdf
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/'docs/collab/20261027_总论文整合';OUT=ROOT/'05_论文/总论文'
DOC=OUT/'2026_MathorCup_大数据竞赛_总论文_v1.docx';N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};W='{'+N['w']+'}'


def main():
    checks=json.loads((AUDIT/'final_checks.json').read_text());assert checks['all_pass']
    manifest=json.loads((AUDIT/'assembly_manifest.json').read_text());pagination=json.loads((AUDIT/'word_pagination.json').read_text(encoding='utf-8-sig'))
    with ZipFile(DOC) as z:root=E.fromstring(z.read('word/document.xml'))
    body=root.find('w:body',N);nodes=list(body);pdf=pymupdf.open(DOC.with_suffix('.pdf'))
    norm=lambda s:re.sub(r'\s','',s)
    texts=[norm(p.get_text()) for p in pdf]
    h1=[h for h in pagination['headings'] if not re.match(r'\d+\.',h['text'])]
    starts=[i for i,x in enumerate(nodes) if x.tag==W+'p' and ''.join(x.xpath('.//w:t/text()',namespaces=N)) in [h['text'] for h in h1] and x.find('w:pPr/w:pStyle',N) is not None and x.find('w:pPr/w:pStyle',N).get(W+'val')=='2']
    chapter=[]
    for j,(index,h) in enumerate(zip(starts,h1)):
        end=starts[j+1] if j+1<len(starts) else next(i for i,x in enumerate(nodes) if '附录（下一轮组装）'==''.join(x.xpath('.//w:t/text()',namespaces=N)))
        last=next(x for x in reversed(nodes[index:end]) if ''.join(x.xpath('.//w:t/text()',namespaces=N)).strip())
        tail=norm(''.join(last.xpath('.//w:t/text()',namespaces=N)))[-18:]
        pages=[i+1-2 for i,t in enumerate(texts) if i>=2 and tail in t]
        endpage=max(pages) if pages else h['body_page']
        chapter.append({'chapter':h['text'],'start':h['body_page'],'end':endpage,'page_count_inclusive':endpage-h['body_page']+1})
    (AUDIT/'chapter_pages.json').write_text(json.dumps(chapter,ensure_ascii=False,indent=2))
    lines=['# 总论文 v1 整合说明','',
    '本稿只整合结构、正文、图表公式编号与版式。题目、摘要、关键词尚为占位，附录未组装；本稿供审阅，未写入06_交付，未提交Git。','',
    '## 产物与版式','',
    '- 总页数25页：物理第1页为题目/摘要/关键词占位，第2页为目录，正文及参考文献22页，末页为附录占位。',
    '- 三问原正文合计36页，整合后的共同前置章节、三问、模型评价及参考文献共22页，少14页，未删三问主图或补检主表。原分问页数包含各自附图和重复内容，压缩比例不作为写作质量指标。',
    '- 14幅主图、20张内容表（含合并符号表）、23个Word原生OMML公式、11条合并文献；27条目录项已由Word实际更新。',
    '- 0043母版先逐字节复制，随后替换内容。styles.xml、numbering.xml、theme、fontTable和settings五部分SHA256与母版完全相同。正文宋体12pt，中文章号/5.1节号、图表标题、三线表及三列公式表沿用母版；表内10pt来自母版实用格式。',
    '- 仅按官方规范增加前置页和正文分节：前两页无页码，正文从1开始、页脚居中阿拉伯页码；无页眉。',
    '- 保留旧正式稿与待审分问稿；未更换模型、训练结果、正式排序或图稿内容。修正了Q2重复描述中的“三感官组”为实际五组15/15/15/4/1。','',
    '## 原章节映射','',
    '| 分问原内容 | 总论文位置 | 处理 |','|---|---|---|',
    '| 三问问题分析、问题背景 | 第1、2章 | 压成背景/任务及三段方法分析，删除重复开篇 |',
    '| 三问模型假设 | 第3章 | 合并六项假设，每项保留依据、失效处理 |',
    '| 分问符号 | 第4章表1 | 合并16行符号表，索引/尺度分开定义 |',
    '| Q1数据与预处理 | 5.1 | 共享主键、250×231/50、缺失与尺度只在此详细解释 |',
    '| Q1模型与求解 | 5.2、5.3 | 基学习器—内层预测—五组NNLS—收缩—截距—外层验证 |',
    '| Q1模型检验及补充检验 | 5.4 | 尾部、风格、配对bootstrap、校准/区间、消融边界均保留 |',
    '| Q2模型/求解/检验 | 6.1、6.2、6.3 | 只引用Q1接口；分开统计排序、模型归因与化学核验 |',
    '| Q3模型/求解/检验 | 7.1、7.2、7.3 | 生成—画像—迁移分层，并列实际结构消融和候选分母 |',
    '| 三问优点、局限、后备方向 | 第8章 | 合并而不重复每问结论 |',
    '| 各问参考文献 | 文末参考文献 | 去重，按首次正文引用连续编号 |',
    '| Q1附图S1—S3、核心代码 | 附录占位 | 按本轮要求不插入；下一轮组装 |','',
    '## 连续编号对照','']
    for kind,title in [('T','表'),('F','图'),('E','公式'),('C','参考文献')]:
        lines += [f'### {title}','','| 来源 | 全文编号 |','|---|---|']
        for source,num in manifest['renumbering'][kind].items():lines.append(f'| {source} | {title}{num} |')
        lines.append('')
    lines += ['符号统一：组收缩λ改为η；Q3路径λ改为t；Copula潜变量z改为u；秩融合S改为H；有符号图拉普拉斯L_s改为L，避免s与风格索引混用。家族索引g与相对等级ĝᵢⱼ按下标和帽符区分。','','## 章节页数','','以下为正文页码。相邻章节可共用一页，因此出现页数不能直接相加。','','| 章节 | 正文页码 | 出现页数 |','|---|---|---|']
    for c in chapter:lines.append(f'| {c["chapter"]} | {c["start"]}—{c["end"]} | {c["page_count_inclusive"]} |')
    lines += ['', '## 典型文字重写','',
    '1. 共享数据口径：原三问分别重复“以beer作为唯一连接键，beer_id不参与建模”等段落；总稿集中在5.1，Q2改为“本问在第5章数据契约与预测接口上识别关联成分”，Q3仅补缺失/地板/参考冲突状态。',
    '2. 模型接口：原Q2再次列四学习器并误述“三个感官组”；总稿不重复NNLS推导，改为“固定第5章同折预测器的基模型、五个感官家族权重和截距，不重新选参或估计NNLS”，同时说明归因并非因果。',
    '3. 结果先行：原Q3“收缩Copula在W1、联合energy和对数离散误差之间取得较均衡的结果”；总稿给出0.4762/0.5813/0.5580/0.2475及整行局部混合0.4740/0.5772/0.6599/0.2925，随后解释覆盖改善与离散误差的权衡，避免“所有变体均更差”。',
    '4. 稳定性集合：将Q2长段汇报拆成bootstrap候选集、正式融合Top-3和总关联参考集三种对象，逐一对应0.6582、0.6076（中位0.6591）及0.6406，不再用一个“留一风格均值”混写。',
    '5. 迁移检验：原稿分散报告早期策略记录和33组任务；总稿并列891=844+47、844=512+205+112+15，明确293为并行标记，再报告33=2+26+1+4及两条候选在0.01/0.02 z裕量下由2降0，区分程序完成与证据支持。','',
    '## 验收结果','',
    '- Word真实TOC已更新，27/27项页码与正文相符；目录在物理第2页，不含自身和附录。',
    '- 原题残留与内部代号：DOCX正文/表格/图题/文献及PDF全文均0命中；0043旧媒体全部移除。数值0.0043是既有标准差，非模板标识，不作伪阳性删除。',
    '- 母版安装标识和用户/设备自定义属性已清空；母版脚注分隔结构保留、旧脚注内容清除。',
    '- 23式逐式渲染目检，空数学run=0；23个式号在PDF均唯一出现，Cambria Math可见；长嵌套求和已消除跨行挤压。',
    '- 20张表不跨页，表题在表前且同页；14幅图与图题同页；相邻公式表之间加入1pt空段，Word识别为独立1×3表。',
    '- 19张分问来源表的数值逐表一致；14幅图按来源次序且图片字节一致。内容来源三份DOCX的SHA256未变，未重跑模型。',
    '- OpenXML SDK验证有效，错误0、警告0；5个母版关键部件逐字节保持。','',
    '## 边界与下一轮事项','',
    '- Q1：10次重复不是独立样本；重复SD区间不是新样本预测区间，残差校准是样本内检查。置换为全局一次目标行置换；小风格只描述Spearman/MAE。部分有序指标及个别目标/风格不占优，均保留。',
    '- Q2：正式统计Top-3与独立归因分开；影子字段占25.4%，非零误选。CO₂在bootstrap第4、最终第3。遮蔽对象是折内条件效应前三字段，不是正式答案，不能称充分性验证。',
    '- Q3：生成仅为训练分布附近的统计画像；未验证未见风格、独立批次及真实酿造。两条候选同源且对代理裕量敏感；小风格相关证据及负对照效应不均匀。',
    '- 下一轮需确定正式题目、摘要和关键词，组装附录核心代码及Q1附图；完成AI使用声明/细节及最终官方提交检查。本轮不替用户提前定稿这些部分。',
    '- 供用户审阅：正文精简程度与三问衔接、目录与章号、图表的总稿尺寸。审阅通过前不覆盖06_交付、不做Git提交。','',
    '## 文件校验','', '| 文件 | 字节 | SHA256 |','|---|---:|---|']
    for name,data in checks['files'].items():lines.append(f'| {name} | {data["bytes"]} | {data["sha256"]} |')
    report='\n'.join(lines)+'\n';(OUT/'整合说明.md').write_text(report);(AUDIT/'整合验收记录.md').write_text(report)
    sums='\n'.join(sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in [DOC,DOC.with_suffix('.pdf'),OUT/'整合说明.md'])+'\n';(OUT/'SHA256SUMS').write_text(sums)
    sheet=Image.new('RGB',(1250,1725),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for i,page in enumerate(pdf):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.38,.38));im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);x=(i%5)*250;y=(i//5)*345;sheet.paste(im,(x+10,y+18));draw.text((x+12,y+3),str(i+1),fill='black')
    sheet.save(AUDIT/'渲染核验/总稿缩略图.png')
    print('Report and SHA256SUMS written.');print(json.dumps(chapter,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
