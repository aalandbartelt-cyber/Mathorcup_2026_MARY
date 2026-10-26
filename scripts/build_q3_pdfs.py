from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/'05_论文/Q3_风格生成'; F=ROOT/'04_图表/Q3_风格生成/draft_v2'
FONT='/home/mary/.local/share/fonts/simfang.ttf'
pdfmetrics.registerFont(TTFont('FangSong',FONT))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CNTitle',fontName='FangSong',fontSize=18,leading=27,alignment=TA_CENTER,spaceAfter=12))
styles.add(ParagraphStyle(name='CNH1',fontName='FangSong',fontSize=15,leading=23,spaceBefore=9,spaceAfter=7))
styles.add(ParagraphStyle(name='CNH2',fontName='FangSong',fontSize=12.5,leading=20,spaceBefore=7,spaceAfter=5))
styles.add(ParagraphStyle(name='CNBody',fontName='FangSong',fontSize=10.5,leading=18,firstLineIndent=21,spaceAfter=6))
styles.add(ParagraphStyle(name='CNNote',fontName='FangSong',fontSize=9,leading=14,alignment=TA_CENTER,spaceBefore=2,spaceAfter=7))
styles.add(ParagraphStyle(name='CNTiny',fontName='FangSong',fontSize=8.5,leading=12))

def P(x,style='CNBody'): return Paragraph(x,styles[style])
def table(data,widths):
 t=Table([[P(str(x),'CNTiny') for x in row] for row in data],colWidths=[w*cm for w in widths],repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9E7F2')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#6E7781')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,1),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
 return t

def footer(canvas,doc):
 canvas.saveState(); canvas.setFont('FangSong',8); canvas.drawCentredString(A4[0]/2,1.0*cm,f'第 {doc.page} 页'); canvas.restoreState()

def main_story():
 s=[P('问题三模型建立与求解','CNTitle'),P('问题三分析','CNH1'),P('问题三要求以啤酒风格为条件生成合理的化合物谱，给出生成样本的感官画像，并将源啤酒迁移到目标风格，同时控制缺陷化合物不恶化。该问同时涉及条件联合分布估计、感官映射和受约束搜索三个环节。','CNBody'),P('数据包含250款啤酒、231个化学及理化字段和22个风格标签；Tripel、Kriek、Lambic分别只有28、10、11个样本。本文保留字段原尺度和观测状态，采用风格内经验边缘与收缩相关结构生成联合谱，并把未知源浓度与未知阈值从缺陷通过分母中分离。','CNBody'),P('问题三模型','CNH1'),P('数据口径与预处理','CNH2'),P('以beer作为唯一连接键，beer_id不参与建模。保留原始缺失；通过正值成对比对识别log10字段和原尺度字段，其余字段保持附件存储尺度。生成与迁移均在训练折内完成分位、尺度、相关阵和边界计算。','CNBody'),table([['对象','处理规则','保留的边界'],['主键','按beer对齐化学谱、风格和感官数据','beer_id不连接、不建模'],['化学字段','按字段级核验结果保留存储尺度','不统一指数化'],['缺失/地板','保留缺失掩码和地板状态','未知浓度不作为安全上限'],['感官画像','使用最终感官模型','0–6仅作经验映射'],['迁移约束','已知直接浓度逐项比较','未知不计入通过分母']],[2.2,6.0,6.0]),Spacer(1,5),P('表1  问题三数据口径与处理规则','CNNote'),P('风格条件收缩Copula','CNH2'),P('将风格内字段按平均秩映射到标准正态空间：zᵢₖ = Φ⁻¹((rᵢₖ−0.5)/nₛ)。在潜在空间中对协方差向总体对角结构收缩，得到相关阵Rₛ；从N(0,Rₛ)采样后按训练风格经验边缘反变换。该模型保留非正态边缘并稳定估计依赖结构。','CNBody'),P('感官画像与相对等级','CNH2'),P('每个生成谱通过第一问最终感官模型得到50维标准化预测，再按训练折经验分布作单调0–6映射。每种风格生成300份谱，图中误差棒表示合成样本第10–90百分位，不是均值置信区间。','CNBody'),table([['维度','Tripel','Kriek','Lambic'],['酒精感','4.4','0.9','1.1'],['苦味','3.9','0.6','1.1'],['酒花味','3.7','0.8','1.6'],['4VG','4.6','0.8','1.3'],['酸度','3.0','5.6','5.6'],['乳酸感','2.9','5.6','5.7']],[3.7,3.0,3.0,3.0]),Spacer(1,5),P('表2  三种风格生成样本的六维相对画像均值','CNNote'),P('受约束迁移','CNH2'),P('源谱x₀到目标端点a沿x(λ)=(1−λ)x₀+λa搜索。对可核验的DMS、双乙酰、4VG和trans-2-nonenal只施加不升约束；未知源值不填补为安全上限。目标接近用目标风格距离衡量，谱改动用标准差归一化RMS衡量。','CNBody'),P('问题三求解','CNH1'),P('求解步骤包括：训练折内拟合边缘和相关结构；统一外层重复比较生成方法；全数据生成三种风格画像；固定候选池和路径搜索迁移；最后按化学可比性、支持域和感官代理稳定性分级。','CNBody'),P('生成质量与模型选择','CNH2'),P('收缩Copula在稳健尺度下的W1为0.4762、联合energy为0.5813、留出覆盖率为0.5580、对数离散误差为0.2475。局部混合的覆盖率更高，但离散误差也增加；基函数邻域混合在Lambic上出现距离退化。因此保留风格条件收缩Copula。','CNBody'),table([['方法','W1↓','energy↓','覆盖率↑','离散误差↓'],['风格收缩Copula','0.4762','0.5813','0.5580','0.2475'],['整行局部混合','0.4740','0.5772','0.6599','0.2925'],['PCA邻域混合','0.4768','0.5805','0.6660','0.3026'],['基函数邻域混合','0.4786','0.5834','0.6580','0.3014']],[4.0,2.3,2.5,2.5,2.5]),Spacer(1,5),P('表3  生成方法的稳健尺度比较','CNNote'),Image(str(F/'图1_生成质量与多样性.png'),width=17.0*cm,height=10.84*cm),P('图1  风格条件生成方法在外层重复验证中的分布保真与多样性','CNNote'),P('三风格感官画像','CNH2'),P('Tripel在酒精感、苦味、酒花味和4VG上较高，Kriek与Lambic在酸度和乳酸感上较高。4VG的外层解释度约为0.255，只作较弱描述证据，画像不能替代真实品评。','CNBody'),Image(str(F/'图2_三风格感官画像.png'),width=17.0*cm,height=9.71*cm),P('图2  三目标风格生成样本的相对感官画像','CNNote'),P('跨风格迁移与缺陷约束','CNH2'),P('33组源—目标任务中，固定最小改动规则得到29条有谱候选：2条进入待实测统计候选，26条直接化学证据不足，1条感官代理不稳定，4组未找到搜索可行谱。两条候选均来自源啤酒230，分别指向Tripel和Kriek；目标距离分别下降38.36%和37.36%，谱RMS改动分别为0.4862和0.5822。两条候选均未取得物理安全认证。','CNBody'),table([['目标风格','目标距离下降','谱RMS改动','代理模型状态','化学状态'],['Tripel','38.36%','0.4862','两模型预测不升','仍有待核验状态'],['Kriek','37.36%','0.5822','两模型预测不升','仍有待核验状态']],[2.7,3.0,2.5,4.3,3.7]),Spacer(1,5),P('表4  两条待实测统计候选的迁移结果','CNNote'),Image(str(F/'图3_迁移收益与审计状态.png'),width=17.0*cm,height=9.51*cm),P('图3  跨风格迁移候选的目标接近程度与证据状态','CNNote'),P('模型检验与结果讨论','CNH1'),P('生成比较采用同一外层重复和折内计算，近训练复制率为0，完整谱重构误差处于10⁻¹²量级。覆盖率提高伴随离散误差增加，说明覆盖不能单独作为生成质量结论。地板感知边缘、共享低秩、神经残差和局部双锚点迁移均未提供足以替换收缩Copula的综合证据。','CNBody'),P('迁移中的“预测不升”只针对冻结感官预测器；Stale hops的外层解释度约0.03、平均绝对误差约0.30 z，预测变化不能直接等同于真实感官改善。已有数据还缺少完整配方、发酵工艺和新实测，因此两条候选应作为待补测的统计设计，不应写成可直接执行的酿造处方或物理安全结论。','CNBody')]
 return s

def build_short(kind):
 if kind=='assump':
  story=[P('问题三模型假设','CNTitle'),P('为简化问题，本文做出以下假设：','CNBody')]
  items=['经字段级核验后，同一化学测量在样本之间具有可比尺度；若字段单位或映射不一致，停止该字段的绝对阈值计算。','风格内主要依赖可由经验边缘与正态Copula近似；若留出联合距离或覆盖不足，保留经验基线并降低该风格结论。','小样本风格的潜在协方差向对角结构收缩；若收缩后相关结构检验变差，如实报告比较结果。','第一问感官预测器可在训练支持域附近提供带误差的画像；若候选离开支持域或模型族方向不稳，则标记为弱证据。','满足已知直接浓度约束的化学谱只能作为统计迁移候选；缺失浓度、检测地板或绝对阈值不可识别时，返回未知。']
 else:
  story=[P('问题三分析','CNTitle'),P('任务本质','CNH1'),P('本问需要同时完成风格条件化合物谱生成、感官画像和跨风格迁移。生成模型要保持边缘分布与化合物间依赖，画像要沿用第一问的感官预测器，迁移则要在目标风格接近与缺陷不恶化之间取得可核验折中。','CNBody'),P('主要难点','CNH1'),P('化学字段维数为231，而三个目标风格样本数仅为28、10、11；缺失、检测地板和参考冲突使部分缺陷浓度不可识别；感官画像和迁移收益又依赖预测器，候选离开训练支持域后均值下降可能掩盖模型不确定性。','CNBody'),P('对策','CNH1'),P('采用风格内经验边缘与收缩相关结构生成联合谱，以同一外层重复协议评价分布距离、覆盖和离散误差；用第一问最终感官模型生成50维画像，并将0–6仅作为经验映射；在固定候选池中沿受约束路径搜索，未知项单列而不计入通过分母。','CNBody')]
  return story
 for i,x in enumerate(items,1): story.append(P(f'假设{i}：{x}','CNBody'))
 return story

def make(path,story):
 doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=1.7*cm,leftMargin=1.7*cm,topMargin=1.6*cm,bottomMargin=1.5*cm)
 doc.build(story,onFirstPage=footer,onLaterPages=footer)

make(Q/'第三问_论文正文_v2.pdf',main_story())
make(Q/'模型假设_Q3_v2.pdf',build_short('assump'))
make(Q/'问题分析_Q3_v2.pdf',build_short('analysis'))
# combined audit PDF: title plus the three independently generated sections.
combined=[P('第三问论文审计合订本','CNTitle'),P('本合订本包含正文、模型假设和问题分析三份交付文档，图稿采用用户已确认的审计版。','CNBody'),PageBreak()]+main_story()+[PageBreak()]+build_short('assump')+[PageBreak()]+build_short('analysis')
make(Q/'Q3_论文审计合订本_v2.pdf',combined)
print('pdfs written')
