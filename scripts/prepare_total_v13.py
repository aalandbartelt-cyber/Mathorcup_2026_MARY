"""Prepare evidence-based manuscript edits; never alter models or result files."""
from copy import deepcopy as cp
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as E
import native_math_total as m
from refine_appendix_v11 import highlight

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261028_V13修订'
BASE = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v12.docx'
NS = {'w': m.W, 'm': m.M}
W = '{' + m.W + '}'
with ZipFile(BASE) as z:
    old = E.fromstring(z.read('word/document.xml'))
body = old.find('w:body', NS)
prototype = body[101]


def pxml(*parts):
    p = E.Element(W + 'p')
    pp = cp(prototype.find(W + 'pPr'))
    for tag in ['keepNext', 'pageBreakBefore', 'numPr', 'sectPr']:
        for element in pp.findall(W + tag):
            pp.remove(element)
    for name in ['firstLine', 'firstLineChars']:
        pp.find(W + 'ind').set(W + name, '480' if name == 'firstLine' else '200')
    p.append(pp)
    for part in parts:
        if isinstance(part, str):
            run = E.SubElement(p, W + 'r')
            rp = E.SubElement(run, W + 'rPr')
            E.SubElement(rp, W + 'rFonts', {W + 'ascii': 'Times New Roman', W + 'hAnsi': 'Times New Roman', W + 'eastAsia': '宋体'})
            E.SubElement(rp, W + 'sz', {W + 'val': '24'})
            E.SubElement(rp, W + 'szCs', {W + 'val': '24'})
            t = E.SubElement(run, W + 't')
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            t.text = part
        else:
            p.append(cp(part))
    return E.tostring(p, encoding='unicode')


def math(*parts):
    return m.math(*parts, size=24)


spec = {
    'base': str(BASE), 'replace_existing': True,
    'replacements': [], 'paragraph_replacements': [], 'insertions': [],
    'global_words': {'审计': '核验', '协议': '设定', '对账': '核对', '冻结': '固定'},
    'reference_map': json.loads((AUDIT / '文献核验.json').read_text())['old_to_new'],
    'references': json.loads((AUDIT / '文献核验.json').read_text())['references'],
    'sections': [],
}

# T5: retain all original numerical facts in their existing table, remove internal prose.
spec['paragraph_replacements'].append({
    'prefix': '早期路径池有891条策略记录',
    'xml': pxml(
        '候选池891条中844条形成化合物谱，其中512条直接证据不足、205条无收益、112条越界被拒；另15条因代理信息缺失暂无法判定，不计为失败。'
        '后续33项迁移任务的两条统计候选，在增加0.01或0.02个标准化感官单位的双模型下降裕量后均不再满足，说明结论接近代理误差边界。'
        '陈旧酒花味代理按10次逐重复指标取均值时，', math(m.sup('R', m.r('2', True))),
        '约0.03、MAE约0.30；图5(c)先平均10次折外预测再评价，分别约为0.06和0.29，二者口径不同。预测不升不能直接替代实测改善。')
})
spec['replacements'] += [
    {'find': '早期分母核验', 'replace': '候选分母核验'},
    {'find': '891记录=844有谱+47结构拒绝；844=512直接证据不足+205无收益+112拒绝+15待代理',
     'replace': '891候选=844有谱+47结构拒绝；844=512直接证据不足+205无收益+112拒绝+15待代理'},
    {'find': 'Pareto标记293为并行属性；15条当时仅等待代理接入',
     'replace': '293条为非支配候选，与上述分类存在重叠；15条因代理信息缺失暂不可判定'},
    {'find': '附录B 补充表格与图片', 'replace': '附录B 补充图片'},
    {'find': '本参赛队在竞赛过程中使用了AI工具，', 'replace': '本参赛队在竞赛过程中使用了AI工具[1][2]，'},
    {'find': '见支撑材料《AI工具使用详情说明.pdf》', 'replace': '见支撑材料《AI工具使用详情说明.pdf》'},
]

# T7: actual split construction, numerical tolerances and candidate tie-breaks.
spec['insertions'].append({
    'prefix': '本文用scikit-learn实现基学习器', 'position': 'after',
    'xml': pxml(
        '外层按tasting_category_fine分层，采用RepeatedStratifiedKFold，参数为n_splits=3、n_repeats=10和random_state=20261023；样本最少的Faro有3款，每个验证折各1款。'
        '以0为起点编号重复和外折，内层采用KFold，设置n_splits=3、shuffle=True，种子为',
        math('20261023+10', m.r('r'), '+', m.r('f')), '，其中', math('r=0,…,9'), '、', math('f=0,1,2'),
        '。基学习器、候选参数与内折的位置均从0计数；随机种子在内层种子上加100000倍基学习器位置、1000倍候选位置及内折位置，完整外层重拟合仅加100000倍基学习器位置。')
})
spec['insertions'].append({
    'prefix': '采用固定10次重复、每次3折', 'position': 'before',
    'xml': pxml(
        '秩高斯变换采用QuantileTransformer(output_distribution="normal", subsample=None)，n_quantiles=min(100, 当前训练样本数)。'
        'PLS2使用scale=True、max_iter=1000、tol=', math(m.sup('10', m.r('−6', True))),
        '。基学习器候选按MAE从小到大选择，同分时依次比较', math(m.sup('R', m.r('2', True))),
        '、Spearman、代理QWK和相邻等级一致率，四项均由大到小；该内层选参顺序与表5中最终结构的综合比较分开。')
})

# T8: actual resampling controls; no additional ranking or fit is executed.
spec['insertions'].append({
    'prefix': '本文将稀疏与图正则回归转为', 'position': 'after',
    'xml': pxml(
        '分层bootstrap共500次，第b次使用种子20261024+10000b（b=0,…,499）；各风格内部有放回抽取原样本数，再合并打乱。'
        '四路先比较Top-10与Top-3支持数，再比较式(13)的倒数秩和，平滑常数为60，同分按最优单路名次及字段代码确定。'
        'CPSS采用100对互补半样本：每个风格内随机排列后对半分，奇数多出的一款随配对编号奇偶轮换；第p对使用NumPy SeedSequence([20261025,p])，p=0,…,99。'
        '负对照随机种子为20261025+117000，包含50次风格内目标置换和20次影子重复，均在相应半样本内重新排序。')
})

# T9: define the actual endpoint rather than inventing a prescribed parameter.
spec['insertions'].append({
    'prefix': '迁移从源谱', 'position': 'before',
    'xml': pxml(
        '迁移端点包括目标风格谱池与同分位端点。目标谱池每风格2000份，按设计尺度到源谱的距离选最近256份；另加入同分位端点',
        math(m.sub('a', 'k'), '=', m.sub('Q', 'sk'), m.delim(m.sub(m.r('F', script='script'), m.r('src,k', True)), '(', ')')),
        '的分量写法：先在源风格其余实测样本中计算源字段的中位秩百分位，再取目标风格同名字段的经验分位，分位数采用线性插值。'
        '此处百分位按小于源值的比例加等于源值比例的一半计算，源啤酒不计入其自身参考分布。')
})
# Replace the temporary compact endpoint equation by a correctly composed inline equation.
endpoint = spec['insertions'][-1]
endpoint['xml'] = pxml(
    '迁移端点包括目标风格谱池与同分位端点。目标谱池每风格2000份，按设计尺度到源谱的距离选最近256份；另加入同分位端点，其第k个分量为',
    math(m.sub('a', 'k'), '=', m.sub('Q', 'sk'), m.delim(m.seq(m.sub(m.r('F', script='script'), m.r('src,k', True)), m.delim(m.sub('x', m.r('0,k', True)))))),
    '。式中，', math(m.sub(m.r('F', script='script'), m.r('src,k', True))),
    '为源风格其余实测样本中小于源值的比例，加上等于源值比例的一半；', math(m.sub('Q', 'sk')),
    '为目标风格同名字段的经验分位函数，采用线性分位插值。源啤酒不计入其自身参考分布。')
spec['insertions'].append({
    'prefix': '本文用经验秩变换及Ledoit-Wolf', 'position': 'after',
    'xml': pxml(
        '生成验证的5个基础种子为20261025、20261026、20261027、20261028、20261029；第r次重复、第f折、第s个风格的实际种子为基础种子+1000r+100f+10s，均从0编号，风格顺序为Tripel、Kriek、Lambic。'
        '每折留出数及每个种子生成数分别为9-10、3-4、3-4，相应训练风格样本为18-19、6-7、7-8，均非单点。'
        '距离与覆盖先在各折、各种子中计算，再求重复内均值、风格等权均值及重复间标准差，不跨折池化后计算距离。'
        '式(17)的收缩强度由各训练风格的Ledoit-Wolf估计自动确定，不设跨风格统一常数。')
})
spec['insertions'].append({
    'prefix': '生成方法复用10×3外层划分', 'position': 'before',
    'xml': pxml(
        '迁移另使用每风格2000份目标谱池，种子为20261025+s；每组采用上述256个邻近端点及1个同分位端点。'
        '路径比例为0.05、0.10、0.20、0.35、0.50、0.75、1.00。主预测器搜索要求目标相对收益至少0.01，代理变化的均值和90%分位均不高于−0.01，支持重复比例至少0.9。'
        '选定最小改动候选后，再由两个预测模型检验零裕量不恶化；表19的0.01/0.02裕量是固定候选的后续敏感性检验，不等同于初选约束。')
})
spec['insertions'].append({
    'prefix': '统计排序冻结后', 'position': 'after',
    'xml': pxml(
        '题面缺陷项的可用阈值取自附件8的Sensory thresholds表threshold_median列：DMS为19 ', math(m.r('μg/L', True)),
        '，双乙酰为0.09 ', math(m.r('mg/L', True)), '，4VG为0.20 ', math(m.r('mg/L', True)),
        '；反-2-壬烯醛的该阈值缺失，记为N/A，不以0替代。直接浓度不升与降至感知阈值以下是两个条件；已知源浓度时可检查前者，阈值缺失不能据此确认后者。')
})
spec['insertions'].append({
    'prefix': '按题面缺陷名单', 'position': 'before',
    'xml': pxml(
        '表16中4VG的4.6是感官相对等级，不能与0.20 ', math(m.r('mg/L', True)),
        '的化学阈值直接比较。图14的实测感官分布也显示Tripel相对较高，但风格特征不免除4VG的缺陷核验；'
        '附件8的校准零存在解释不确定性，不能据此认定真实浓度为0或不存在越阈。')
})

# T11: explain the trade-off without inventing a retrospective preregistration.
spec['insertions'].append({
    'prefix': '连续-有序融合的MAE为0.3356', 'position': 'after',
    'xml': pxml(
        '最终选择侧重连续感官信号的解释度，同时检查尾部目标和风格稳健性：表5中感官组收缩NNLS的',
        math(m.sup('R', m.r('2', True))),
        '最高，而连续-有序融合的四项误差或排序指标略优，故不将最终方法表述为单项全面占优。'
        '这些差异来自同一组外层划分的方法比较，不是独立最终测试；表6只比较最终模型与基线，不能据此判定其相对连续-有序融合的四项微小差异显著。')
})

# Real source excerpts, exactly as shipped. Metadata headers are outside the ranges.
mapping = json.loads((AUDIT / '附录14节_真实源范围.json').read_text())
for letter, purpose in [('C', '问题一感官组收缩NNLS与训练折内求解'),
                        ('D', '问题二稳健排序、条件归因与化学核验'),
                        ('E', '问题三风格生成、画像预测与约束迁移')]:
    question = {'C': '一', 'D': '二', 'E': '三'}[letter]
    section = {'letter': letter, 'heading': f'附录{letter} 问题{question}模型核心程序',
               'intro': f'本附录列出{purpose}的核心函数节选，完整源程序见表A.2。以下命令在解压后的A2601605附件目录执行。',
               'blocks': []}
    for b in mapping['sections']:
        if not b['section'].startswith(letter):
            continue
        source = '\n'.join(b['source_lines'])
        assert not re.search(r'\bE\d{3}\b|BZD|agent|学长版|SHA256|Source logical name', source)
        versions = 'Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、scikit-learn 1.7.2'
        section['blocks'].append({
            'title': f'{b["section"]} {b["title"]}（{b["file"]}）',
            'path': b['archive_relative_path'], 'source_lines': b['source_lines'],
            'lines': highlight(source),
            'software': f'软件：{versions}。语法检查命令：{b["command"]}。',
            'context': [b['location_note']],
        })
    spec['sections'].append(section)

spec['table14_rows'] = [
    ['校准零', '保留原始/校准尺度的零状态及其来源', '不把校准零直接认作真实无浓度'],
    ['参考冲突', '参考字段与输入不一致时保留冲突记录', '不静默替换实测输入或认定安全']]

(AUDIT / '修订规格.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
print('Prepared V13 edits:', len(spec['insertions']), 'insertions,',
      sum(len(s['blocks']) for s in spec['sections']), 'real source excerpts')
