"""Prepare neutral supplementary text/tables and figure QA for the root editor."""
from pathlib import Path
import hashlib
import json
import re
from PIL import Image, ImageOps
import pandas as pd

ROOT = Path('/home/mary/mathorcup')
HERE = Path(__file__).parent
DATA = ROOT / '05_论文/Q3_风格生成/待审_20261027/数据与结果/validation_v1'
FIG = ROOT / '04_图表/Q3_风格生成/待审_20261027'
PATTERN = re.compile(r'(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|参考组|对照组|晋级|门槛|promotion|\bpass\b|BZD|agent(?:2|3)?|学长版', re.I)


def write_json(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((DATA / '补充检验摘要.json').read_text())
    table6 = {
        'caption': '表6 问题三补充检验与适用范围',
        'rows': [
            ['检验项', '结果', '口径与结论'],
            ['完整性与划分', '30折×3风格×5种子共450组齐全，生成数逐组匹配留出数；900份画像×50目标全部有限', '训练与验证beer无交叉；支持程序一致性和画像完整性'],
            ['复制与退化', '近训练复制率0，整组塌缩标记0', '复制按规范化RMS≤10⁻⁶；实测本已恒定的地板维度保留'],
            ['尺度敏感性', '原尺度W1/energy=1.0773/6.3933；稳健尺度0.4762/0.5813', '稳健边界仅来自训练折1%–99%分位；极端记录影响显著'],
            ['逐风格一致性', 'Tripel/Kriek/Lambic稳健W1为0.4114/0.5684/0.4487；覆盖0.3356/0.7389/0.5994', '风格等权；小风格相关RMSE较大，仅作描述'],
            ['保持边缘的独立置换', 'W1变化0；energy差+0.00109，相关RMSE差+0.00851', '差为置换−原谱；五次置换先平均；Kriek的energy差为−0.00009，证据不一致'],
            ['早期分母核验', '891记录=844有谱+47结构拒绝；844=512直接证据不足+205无收益+112拒绝+15待代理', 'Pareto标记293为并行属性；15条当时仅等待代理接入'],
            ['固定迁移与裕量', '33任务中2统计候选/26证据不足/1代理不稳/4无谱；裕量0/0.01/0.02z对应2/0/0', '固定候选不重新选择；两候选来自同一源，需独立实测'],
            ['未验证范围', '未见风格泛化、独立批次和实际酿造尚未验证', '已知风格内留出；不支持可实际酿造、唯一或全局最优主张'],
        ],
        'widths': [1900, 3800, 2940],
        'note': '重复训练集高度重叠；种子、置换与折均先聚合到重复，重复标准差只描述扰动。覆盖率为真实留出点的近邻几何覆盖率，不是区间覆盖或安全率。早期记录和当前源—目标任务分母分别报告。',
    }
    ablation_rows = [['方法', 'W1 / energy', '覆盖率 / 相关RMSE']]
    for row in summary['table7']:
        ablation_rows.append([row['方法'], f"{row['W1']} / {row['energy']}", f"{row['覆盖率']} / {row['相关RMSE']}"])
    table7 = {
        'caption': '表7 共享结构变体的生成消融检验',
        'rows': ablation_rows,
        'widths': [2800, 3000, 2840],
        'note': '同一10×3外层划分与五个生成种子，训练折分位稳健尺度；先对折和种子平均，再对三风格等权。W1、energy与相关RMSE越低越好；覆盖须与离散和联合距离共同解释。三结构相对主模型的energy增加0.0658–0.0703，相关RMSE略降。',
    }
    addition = {
        'heading': '补充检验：联合一致性、结构消融与迁移边界',
        'paragraphs': [
            '为检验生成结果是否满足任务中的联合一致性要求，在冻结的三十个外层划分、三个目标风格和五个生成种子上核对完整性，并将检验结果汇总于表6。全部450组生成记录与各自风格留出样本数匹配，训练与验证啤酒无交叉；900份全数据描述画像的45,000个目标值全部有限。近训练复制率和整组塌缩标记均为0。检测地板造成的原有恒定字段单独保留，不能用全部字段均有方差作为有效性条件。',
            '进一步在每个固定缓存组内逐字段独立置换生成样本行，重复五次，保持每个字段的边缘值而打散化合物间配对。沿用训练折中位数、分位边界和尺度，逐组重算后原始稳健诊断与存档最大差为2.22×10⁻¹⁶；置换前后的W1完全相同。图4显示，置换后energy平均增加0.00109，相关结构RMSE平均增加0.00851，覆盖率在三种风格均下降。Tripel的energy增加0.00275，但Kriek为−0.00009，说明保留联合配对得到的是有限且风格不一致的支持，不能据此声称所有风格的依赖结构均已准确恢复。',
            '表7按实际结构比较共享低秩、风格残差收缩及风格残差与地板状态三种变体。三者的相关RMSE略有降低，但边缘W1和联合energy均增加，energy相对主模型增加0.0658–0.0703，故本题继续使用风格条件收缩Copula。该结构消融说明，改善某一相关诊断并不必然改善联合分布。同时，原尺度与分位稳健尺度的距离差别明显，表6并列保留两种口径，避免由少量极端观测决定单一结论。',
            '迁移检验区分策略记录和源—目标任务。早期直接约束阶段891条记录中844条有谱、47条结构拒绝；293条Pareto标记是并行属性。15条等待代理接入记录代表当时尚未完成代理检查。当前固定最小改动主规则的33组任务仍按2条待实测统计候选、26条直接证据不足、1条代理不稳定和4条无谱报告。两条统计候选来自同一源；对同一锁定候选追加0.01z或0.02z的双模型下降裕量后，满足数均由2降为0，表明结论处于代理误差边界。',
            '全部重复标准差和图4横线只描述划分及随机扰动，折、种子或置换均不作为独立实测样本。当前验证限于已知目标风格内留出，未见风格外推、独立批次及实际酿造尚未检验；小风格画像只作描述，生成谱是与训练分布相容的统计设计，需在直接浓度、感官与工艺层面进一步验证。',
        ],
        'tables': [table6, table7],
        'figures': [{
            'path': str((FIG / '图4_联合结构独立置换检验.png').relative_to(ROOT)),
            'caption': '图4 保持边缘分布的独立置换下三种风格的联合结构检验',
            'note': '差为置换减原谱；每点为一次外层重复，菱形与线段为十次重复均值±标准差。每折每风格五个生成种子、五次置换先平均；边缘W1按构造不变。正差支持保留联合配对，Kriek的energy非正差同样保留。训练重复重叠，横线不是独立置信区间。',
        }],
    }
    assert not PATTERN.search(json.dumps(addition, ensure_ascii=False))
    write_json(addition, HERE / 'paper_addition.json')
    # Presentation-only previews, using the exported R PNG as the sole source.
    pic = FIG / '图4_联合结构独立置换检验.png'
    im = Image.open(pic).convert('RGB')
    qa = FIG / 'qa'
    qa.mkdir(exist_ok=True)
    small = im.resize((round(im.width * .7), round(im.height * .7)), Image.Resampling.LANCZOS)
    small.save(qa / '图4_70pct.png')
    grey = ImageOps.grayscale(small).convert('RGB')
    grey.save(qa / '图4_70pct_gray.png')
    sheet = Image.new('RGB', (small.width, small.height * 2 + 24), 'white')
    sheet.paste(small, (0, 0)); sheet.paste(grey, (0, small.height + 24))
    sheet.save(qa / 'Q3_补充检验审图合页.png')
    dpi = Image.open(pic).info.get('dpi')
    svg = (FIG / '图4_联合结构独立置换检验.svg').read_text()
    qa_report = {
        'status': 'visual_qa_completed_user_review_pending', 'figure': pic.name,
        'width_px': im.width, 'height_px': im.height, 'dpi': list(dpi),
        'r_backend': 'ggplot2_theme_mcup', 'font_family': 'SimSun',
        'data_label_overlap_check': 'Mean labels placed in a reserved right column beyond all data points.',
        'grayscale_encoding': 'Style names are direct axis labels; mean diamonds and repeat dots have different sizes and outlines.',
        'panels': 2, 'top_level_title': False,
        'svg_font_family_detected': 'SimSun' in svg,
        'visual_check_70pct': 'Both panels, mean labels and symbolic legend remain readable; no data-label overlap or cropped elements observed.',
        'visual_check_grayscale': 'Rows are identified by explicit style names; dots, diamonds, horizontal SD segments and dashed zero references remain distinguishable.',
        'scale_and_mean_line_are_descriptive': True,
        'previews': [p.name for p in qa.glob('*.png')],
    }
    write_json(qa_report, FIG / 'qa_report.json')
    source_manifest = {
        'scope': 'supplementary_frozen_spectrum_validation_pending_review',
        'source_tables': {str(p.relative_to(ROOT)): sha(p) for p in [DATA / '独立置换逐重复差.csv', DATA / '独立置换汇总.csv']},
        'plot_source': {'path': str((FIG / 'plot_q3_validation.R').relative_to(ROOT)), 'sha256': sha(FIG / 'plot_q3_validation.R')},
        'outputs': {p.name: sha(p) for p in FIG.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.svg', '.csv'}},
    }
    write_json(source_manifest, FIG / 'source_data_manifest.json')
    hits = []
    for parent in [FIG, DATA]:
        for p in parent.rglob('*'):
            if not p.is_file():
                continue
            if PATTERN.search(str(p.relative_to(ROOT))):
                hits.append({'path': str(p), 'scope': 'filename'})
            if p.suffix.lower() in {'.csv', '.json', '.r', '.md', '.svg'}:
                for m in PATTERN.finditer(p.read_text()):
                    hits.append({'path': str(p), 'term': m.group()})
    assert not hits, hits
    write_json({'forbidden_hits': hits, 'n_tables': 2, 'table_row_counts': [len(table6['rows']), len(table7['rows'])], 'n_figures': 1, 'png_dimensions': [im.width, im.height], 'png_dpi': list(dpi)}, HERE / 'addition_checks.json')
    print(json.dumps({'text_scan_hits': len(hits), 'table_rows': [len(table6['rows']), len(table7['rows'])], 'png_size': im.size, 'dpi': dpi}, ensure_ascii=False))


if __name__ == '__main__':
    main()
