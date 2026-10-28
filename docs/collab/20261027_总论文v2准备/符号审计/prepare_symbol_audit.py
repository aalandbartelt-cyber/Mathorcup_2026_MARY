"""Read-only manuscript/code audit; writes audit evidence, never edits a paper."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import csv,json,re
from lxml import etree as E

ROOT=Path('/home/mary/mathorcup')
OUT=Path(__file__).resolve().parent
DOC=ROOT/'05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v1.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}

# Scope-aware concepts. Composite vectors and their components form one family;
# different objects sharing a glyph are explicitly kept as separate rows.
DATA=[
('i','i','啤酒样本索引','1…N，N=250',1,'已在符号章介绍；模型首次处未就地复述','Eq1后定义样本、输入和输出'),
('k','k','化学及理化字段索引','1…p，p=231；Q2可解释资格集合218',10,'仅符号章概括','区分字段资格与模型输入维数'),
('j','j','感官目标索引','1…J，J=50',2,'仅符号章概括','Eq1后预先定义J和j'),
('m','m','基学习器索引','1…M，M=4',2,'模型前段提及第m个，未说明范围','Eq2后定义M=4'),
('s','s','啤酒风格索引','全体22；生成重点三种风格',16,'符号章介绍，模型中未就地重述','Eq16后就地定义'),
('g','g','感官家族索引','5组，15/15/15/4/1维',3,'正文已列五组但未定义g(j)','Eq3解释g；Eq4定义g(j)'),
('G','G_g','第g个感官家族的目标下标集合','G_g为{1,…,J}的预定义子集',2,'公式前已解释G为集合','将G统一写G_g，避免G/g大小写关系含混'),
('N','N','一次外层重复完整OOF评价的样本数','N=250，无量纲计数',7,'v1使用n而未定义评价分母','Eq7/8用N，避免与训练折样本数n_T混用'),
('n_train','n_T','当前训练子集样本数','n_T=|T|，不是固定250',10,'v1 n在相关/回归/OOF评价漂移','相关/回归式用n_T并分别说明作用域'),
('J','J','感官目标总数','J=50',7,'无模型首次解释','Eq1后或Eq7后定义'),
('p','p','保留原始存储字段的谱向量维数','p=231',17,'v1直到式23都未明确p','Eq1后或Eq17后定义'),
('q','q','Q2预处理回归设计矩阵列数','231原字段+训练折缺失指示，随折改变',11,'v1缺失，不能写218','新增以忠实表述实际模型输入'),
('T_train','T','当前外层训练啤酒下标集合','T不包含外层验证观测',2,'公式前有定义','保留，并将bootstrap集合T_b^(j)改S_b^(j)'),
('x','x_i,x_{ik}','第i款啤酒输入谱及第k字段','R^231；各字段原存储尺度，含log10或原尺度',1,'公式前有化学向量定义','补维度及单位按字段确定，迁移设计坐标改v'),
('y','y_i,y_{ij},y_j','标准化感官向量、单元与第j目标列向量','50维或对应n行；标准化z尺度',1,'向量与标量未完整区分','Eq1/2与Eq11分别说明行向量/标量/目标列'),
('F','F,F_j','化学谱到感官向量的映射及第j分量','R^p→R^J，目标值为标准化连续量',1,'只给映射维度未说明分量','Q2的f_j可统一为F_j，避免与基学习器f_m混淆'),
('epsilon','epsilon_i,epsilon_{ij}','不可解释残差向量/单目标残差','与对应标准化感官值同尺度',1,'公式前泛称误差项','Eq1后明确ε_i向量；Q2分量为ε_ij'),
('z_stack','z_{ijm}','训练内层3折的折外基学习器预测','训练样本×50目标×4模型；标准化感官尺度',2,'式前已定义','保留并将晚文中的0.01z单位改普通“个标准差单位”，避免z另义'),
('ybar_train','bar y_{j,T}','第j目标当前训练集均值','标准化感官尺度',2,'式中未定义bar的集合','加T下标；评价均值用bar y_{j,eval}'),
('zbar_train','bar z_{jm,T}','第j目标第m学习器训练内层OOF均值','同z_{ijm}尺度',2,'未定义','Eq2后定义并全文统一T范围'),
('sigma_target','sigma_{j,T}','当前训练目标总体标准差','ddof=0；≤1e-12按1处理',2,'v1 s_j在式前提及且只写零标准差','改σ_{j,T}避风格s；阈值按代码，不能写仅严格为0'),
('w_optim','w_g^*,w_all^*','家族/全部目标的未收缩非负NNLS解','R_+^4；和不要求为1',2,'式前未全定义','Eq2后定义局部解，Eq3后定义共享解'),
('w_final','tilde w_g,tilde w_{g(j),m}','最终组收缩权重向量及第m分量','R_+^4；0.5收缩',3,'只笼统称家族解','Eq3后定义最终解，与未收缩w*区分'),
('eta','eta','固定家族向共享权重的收缩系数','η=0.5，无量纲',3,'公式前已定义','保留，不重复定义'),
('group_map','g(j)','第j感官目标所属家族的映射','值为五家族之一',4,'正文未定义','Eq4后补定义'),
('intercept','b_j','第j目标的自由截距','标准化感官尺度；可负',4,'公式前已定义','保留，补与训练均值恢复的关系'),
('base_function','f_{mj}^{(T)}','第m基学习器在T重拟合后的第j输出','输入R^p，输出单目标标量',5,'v1写f_m^(T)，多输出/标量分量含混','显式加j分量，不改变组合算法'),
('yhat','hat y_{ij}','对外层验证样本的最终连续预测','标准化感官尺度',5,'公式前泛称预测但未说明帽符','Eq5后就地定义'),
('cdf','hat C_{j,T}','第j目标由训练集拟合的经验分布函数','实数→[0,1]',6,'v1 F_j,tr与F预测器复用F','建议换C，不改变映射'),
('grade','hat q_{ij}','第i款啤酒第j目标的相对强度等级','0…6，非原始评分',6,'v1 hat g_ij与家族g复用','改qhat或另未占用符号'),
('ybar_eval','bar y_{j,eval}','第j目标当前重复全体OOF真值均值','N=250完整评价集合',7,'v1 bar y_j与训练均值混用','与bar y_{j,T}分开，禁止训练均值替代评价均值'),
('r2_macro','R_macro^2','逐目标R²的等权平均','无量纲；与mean-OOF评价分开',7,'中文解释已有，分母n/J未定义','明确N、J及评价观测集合'),
('mae_macro','MAE_macro','逐目标绝对误差的等权均值','1/(NJ)求和，标准化感官尺度',8,'名称已定义，n/J未定义','函数名MAE及macro下标用正体'),
('rank_chem','R^X_{ik},bar R^X_k','化学字段在当前相关分析样本中的平均秩及均值','1…n_T，并列取平均秩，无量纲',10,'公式前仅给R_ik','加X标识，避免与目标秩/风格相关矩阵R_s混淆'),
('rank_sens','R^Y_{ij},bar R^Y_j','感官目标在当前相关分析样本中的秩及均值','1…n_T，并列取平均秩，无量纲',10,'公式前仅给R_ij','加Y标识'),
('rho','rho_{kj}','字段k与目标j的Spearman秩相关','[-1,1]；常值情形按实现单列处理',10,'未就近定义希腊符号','Eq10后定义'),
('X_design','X','折内预处理后的回归设计矩阵','R^{n_T×q}，231字段+训练折指示',11,'v1未定义X且易误读218列','Eq11后定义，保留缺失指示图惩罚拓展口径'),
('beta','beta,hat beta_j','当前目标待估系数及最小化所得系数','R^q；可正可负',11,'无首次定义且β/β_j混写','目标右侧所有待估变量统一β；左侧hatβ_j'),
('L','L','有符号归一化图拉普拉斯，拓展至q列','R^{q×q}半正定；指示列不连图边',11,'式前已提名称，但矩阵范围未解释','保留L；不用L_s的signed下标'),
('lambda1','lambda_1','L1稀疏惩罚系数','正数，无量纲调参',11,'式前已定义','保留'),
('lambda2','lambda_2','L2收缩惩罚系数','非负；实际代码启用',11,'v1公式完全遗漏该项','补λ2||β||_2²，与实际GraphElasticNet对齐'),
('lambdaG','lambda_G','图二次惩罚系数','非负；训练内选取',11,'式前已定义','G为graph的正体说明下标，不与家族集合G_g混淆'),
('delta_cond','Delta_{kj}^{cond}','固定预测器中第k字段条件置换引起的第j目标MAE增量','标准化感官尺度，正表示误差升高',12,'式前解释正方向但未完整解释输入替换','cond正体；不是因果效应'),
('X_minus_k','X_{-k},tilde x_k','保留其余输入与按条件方案替换的第k字段','与X同一折预处理范围；保留缺失掩码',12,'未定义','Eq12后解释，不能称删除后重训'),
('H','H_{kj}','第k候选对第j目标的倒数秩融合分数','四路sum 1/(60+r)，无量纲',13,'式前定义名词但未说明60','Eq13后定义60为固定平滑常数'),
('ell','ell','证据路索引','1,2,3,4',13,'未定义','Eq13后列四路含义及顺序'),
('r_route','r_{ell,kj}','候选k对目标j在第ell路的名次','从1开始的正整数；按已冻结排名规则',13,'泛称名次，未说明三索引','Eq13后明确三个索引'),
('pi','pi_{kj}','bootstrap入选前三的频率','[0,1]',14,'式前已定义','保留'),
('B','B','bootstrap重抽次数','B=500',14,'式前已提500但B未对应','Eq14后明确B=500'),
('b_boot','b','bootstrap重复索引','1…B',14,'未定义；与b_j截距同字母','独立索引b可保留，需明确不是截距；亦可用r_boot避免'),
('S_boot','S_b^(j)','第b次bootstrap第j目标入选Top-3字段集合','218严格单体中的3项集合',14,'v1写T_b^j，未定义且T已是训练集','改S_b^(j)，下标/上标都为真OMML'),
('indicator','I(·)','事件成立取1否则取0的示性函数','{0,1}',14,'符号表有，模型未解释','I用正体，非单位矩阵I_p'),
('OAV','OAV_k','字段k无量纲气味活性值','浓度/感知阈值，仅同字段同单位可比',15,'式前已定义','保留正体函数缩写'),
('concentration','c_k,c_k(x)','字段k校准浓度或由谱x按已知尺度还原的浓度','各化合物原单位；原尺度或10^存储值',15,'c_k有，c_k(x)未定义函数','Eq15定义浓度，Eq21补函数含义'),
('tau','tau_k','字段k对应感知阈值','须与c_k同单位且>0',15,'式前已定义','保留'),
('rank_style','r_{ik}^{(s)}','风格s内第k字段的样本平均秩','1…n_s，并列平均秩',16,'式前有r_ik但未标风格','加s可选，至少就地明确风格内秩'),
('n_s','n_s','当前训练风格s的样本数','不是固定Tripel28/Kriek10/Lambic11全量值',16,'式前已定义','保留，强调折内值'),
('u_latent','u_{ik},u_k^*','正态Copula的训练潜在变量及生成潜在分量','无量纲正态坐标；R^p',16,'式前定义uik，生成u*未定义','Eq19后解释抽样潜在向量'),
('Phi','Phi,Phi^{-1}','标准正态分布函数及其反函数','实数↔(0,1)',16,'未定义','Eq16后定义，函数名正体'),
('Sigma','Sigma_s,hat Sigma_s','风格潜在数据MLE协方差及Ledoit-Wolf收缩协方差','p×p，无量纲',17,'正文提名称但帽符/估计分母不清','Eq17后定义二者及trace,p'),
('alpha','alpha_s','Ledoit-Wolf球形收缩强度','[0,1]，当前训练风格估计',17,'未定义α_s','Eq17后定义'),
('I_matrix','I_p','p阶单位矩阵','p×p；区别于I(事件)',17,'未定义','加p阶下标，与示性函数区分'),
('trace','tr(·)','矩阵迹，即对角元素之和','标量；除p得平均方差',17,'未解释','常见算子可不入全局表，但首次式后解释'),
('D','D_s','hatΣ_s对角元素构成的对角矩阵','p×p；零方差数值保护按代码',18,'式前有名称，缺求逆/数值保护说明','Eq18后说明D_s^{-1/2}逐对角取倒平方根'),
('R_copula','R_s','生成用正态Copula潜在相关矩阵','p×p；非迁移设计空间精度',18,'式前定义但式22滥用','仅用于生成，不用于迁移距离'),
('Q','Q_{sk}','风格s字段k的训练经验分位函数','[0,1]→该字段存储尺度；线性分位插值',19,'式前已提','Eq19补反变换/线性插值'),
('xstar','x_k^*','第k字段生成存储值','与训练字段同尺度',19,'未就近解释*标志','Eq19后明确星号是生成样本非最优解'),
('x0','x_0','源啤酒谱','R^p，附件存储尺度',20,'公式前已定义','保留；其设计坐标为v_0'),
('anchor','a','目标风格候选端点谱','R^p，同源谱尺度',20,'公式前已定义','保留，不称唯一最优'),
('t','t','从源谱指向目标端点的路径比例','[0,1]，无量纲',20,'公式前已定义','保留'),
('x_path','x(t)','按t线性插值的候选存储谱','R^p；不是物理浓度线性混合',20,'未定义函数名','Eq20后明确log字段插值不等于原浓度线性混配'),
('cstar','c_k^*(t)','已知直接约束下投影后的浓度','同c_k；未知状态不作已知上界',21,'式前泛称不升约束','Eq21后定义直接项目集合和未知规则'),
('v_design','v(x),v_0','按固定设计参考均值/标准差标准化的化学坐标','R^p，单位为字段标准差',22,'v1复用x但仅文字提醒','改显式v，避免x跨尺度'),
('center_scale','mu_x,sigma_x','设计参考集各字段均值/标准差','原存储字段尺度；≤1e-12替1',22,'v1未写标准化变换','就地说明v_k=(x_k−mu_x,k)/sigma_x,k'),
('mu_style','mu_s^v','目标风格在设计坐标中的均值','R^p，无量纲',22,'未定义且与潜在均值易混','显式加v上标，与Copula协方差区分'),
('Omega','Omega_s','设计坐标中Ledoit-Wolf协方差的精度矩阵','p×p，无量纲，来自TargetGeometry.precision_',22,'v1误用Copula R_s^{-1}','新增Ω_s，换成代码实际sqrt(δᵀΩδ/p)'),
('distance','d_s(x)','源/候选到目标风格设计均值的归一化Mahalanobis距离','sqrt(二次型/p)，无量纲',22,'v1漏sqrt和/p','修正文档公式，保留所有已有距离和收益数值'),
('Delta_RMS','Delta(x,x_0)','候选相对源的标准化谱RMS改动','sqrt(sum((v_k−v0k)^2)/p)，无量纲',23,'v1用x后文字说设计坐标，符号仍二义','改显式v或写((xk−x0k)/sigma_x,k)')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ZipFile(DOC) as z:root=E.fromstring(z.read('word/document.xml'))
    body=root.find('w:body',NS);eq=[];number=0
    for i,x in enumerate(body):
        maths=x.xpath('.//m:oMath',namespaces=NS)
        if maths:
            number+=1;eq.append({'equation':number,'body_child_index':i,'flat_diagnostic_only':''.join(x.xpath('.//m:t/text()',namespaces=NS)),'xml':E.tostring(maths[0],encoding='unicode')})
    equations={e['equation']:e for e in eq}
    rows=[]
    for key,symbol,meaning,unit,first,status,action in DATA:
        rows.append({'id':key,'symbol_v2_recommended':symbol,'meaning':meaning,'unit_or_range':unit,'first_formula':first,'first_body_child_index':equations[first]['body_child_index'],'first_section':'5.2' if first<=6 else '5.3' if first<=8 else '6.1' if first<=15 else '7.1','v1_definition_status':status,'action':action})
    with (OUT/'symbols.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'symbols.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (OUT/'v1_equations.json').write_text(json.dumps(eq,ensure_ascii=False,indent=2))
    unicode_rows=[]
    math_char=re.compile(r'[₀-₉ₐ-ₜᵢⱼₖₘₙₛᵣᵤᵥᵦ-ᵨ⁰-⁹α-ωΑ-Ω]|R²|\bRMS\b|\bMAE\b|\bRMSE\b')
    for i,x in enumerate(body):
        texts=x.xpath('.//w:t/text()',namespaces=NS);value=''.join(texts)
        hits=math_char.findall(value)
        if hits:
            unicode_rows.append({'body_child_index':i,'text':value,'tokens':hits,'note':'仅数学语境替换为行内OMML；图内既有栅格/正式指标名与文献题名不可盲目替换'})
    (OUT/'正文Unicode数学替换清单.json').write_text(json.dumps(unicode_rows,ensure_ascii=False,indent=2))
    symboltable=next(t for t in body.findall('w:tbl',NS) if '全文主要符号' in ''.join(t.getprevious().xpath('.//w:t/text()',namespaces=NS)))
    symbolcells=[tr.findall('w:tc',NS)[0] for tr in symboltable.findall('w:tr',NS)[1:]]
    facts={'source':str(DOC),'source_sha256':sha256(DOC.read_bytes()).hexdigest(),'scope_aware_symbol_concepts':len(rows),'display_equations':number,'v1_notation_rows':len(symbolcells),'v1_symbol_column_math_count':sum(len(c.xpath('.//m:oMath',namespaces=NS)) for c in symbolcells),'paragraphs_or_cells_with_unicode_math':len(unicode_rows),'code_evidence':{}}
    for p in [ROOT/'.worktrees/q1-nnls-opt/02_代码/Q1_感官评分/src/nnls_stacking.py',ROOT/'.worktrees/q1-nnls-opt/02_代码/Q1_感官评分/src/q1_metrics.py',ROOT/'.worktrees/q2-evidence-opt/02_代码/Q2_关键化合物/src/selection_models.py',ROOT/'.worktrees/q2-evidence-opt/02_代码/Q2_关键化合物/src/run_e101_e104.py',ROOT/'.worktrees/q3-foundation/02_代码/Q3_风格生成/src/run_e204_constrained_migration.py',ROOT/'.worktrees/q3-foundation/02_代码/Q3_风格生成/src/gaussian_baselines.py']:
        facts['code_evidence'][str(p.relative_to(ROOT))]=sha256(p.read_bytes()).hexdigest()
    (OUT/'审计摘要.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
    print(json.dumps(facts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
