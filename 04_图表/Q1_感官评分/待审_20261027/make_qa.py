"""Create only derivative QA previews for the three R-generated figures."""
from pathlib import Path
from hashlib import sha256
import json
from PIL import Image,ImageOps
import numpy as np
import pymupdf

OUT=Path(__file__).resolve().parent
QA=OUT/'qa';QA.mkdir(exist_ok=True)
stems=['图5_重复验证稳定性与预测校准','附图S2_消融与负对照','附图S3_逐目标预测可靠性']
records=[];contacts={'color':[],'gray':[]}
for stem in stems:
    src=OUT/(stem+'.png');im=Image.open(src).convert('RGB')
    pix=np.array(im);mask=np.any(pix<245,axis=2)
    yy,xx=np.where(mask)
    margin=[int(xx.min()),int(yy.min()),int(im.width-1-xx.max()),int(im.height-1-yy.max())]
    small=im.resize((round(im.width*.7),round(im.height*.7)),Image.Resampling.LANCZOS)
    small.save(QA/(stem+'_70pct.png'),dpi=(300,300))
    gray=ImageOps.grayscale(im).convert('RGB');gray.save(QA/(stem+'_gray.png'),dpi=(300,300))
    gray.resize(small.size,Image.Resampling.LANCZOS).save(QA/(stem+'_gray_70pct.png'),dpi=(300,300))
    for label,view in [('color',im),('gray',gray)]:
        contacts[label].append(view.resize((1000,round(view.height*1000/view.width)),Image.Resampling.LANCZOS))
    pdf=pymupdf.open(OUT/(stem+'.pdf'));txt='\n'.join(p.get_text() for p in pdf)
    records.append({'stem':stem,'pixels':[im.width,im.height],'dpi':list(Image.open(src).info.get('dpi',[])),
                    'white_outer_margins_px':margin,'pdf_pages':len(pdf),'pdf_sha256':sha256((OUT/(stem+'.pdf')).read_bytes()).hexdigest(),
                    'png_sha256':sha256(src.read_bytes()).hexdigest(),'pdf_text_characters':len(txt)})
for label,ims in contacts.items():
    sheet=Image.new('RGB',(sum(x.width for x in ims)+40,max(x.height for x in ims)+20),'white')
    x=10
    for im in ims:sheet.paste(im,(x,10));x+=im.width+10
    sheet.save(QA/f'Q1_validation_{label}_contact.png',dpi=(300,300))
(OUT/'figure_qa.json').write_text(json.dumps({'records':records,'manual_review':'pending root/user review; derivatives and geometric margins alone do not prove absence of overlap'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
