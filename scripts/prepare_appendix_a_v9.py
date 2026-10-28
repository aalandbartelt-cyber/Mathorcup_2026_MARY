"""Prepare factual Appendix A tables from the Windows submission folder."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT = Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/A2601605附件')
AUDIT = ROOT / 'docs/collab/20261028_总论文v9附录A'


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_folder = ATTACHMENT / 'A2601605源程序'
    source_paths = sorted(p for p in source_folder.iterdir() if p.is_file())
    assert len(source_paths) == 30
    assert Counter(p.suffix for p in source_paths) == {'.py': 20, '.R': 10}
    expected = {
        'A2601605源程序': 30,
        'A2601605数据_不含赛题原始数据': 18,
        'A2601605结果': 111,
        'A2601605论文图表': 21,
    }
    for name, count in expected.items():
        assert len([p for p in (ATTACHMENT / name).iterdir() if p.is_file()]) == count
    txt = ATTACHMENT / 'A2601605软件与运行说明.txt'
    assert txt.is_file()
    overview = [
        ['序号', '文件名称', '格式', '主要内容'],
        ['1', 'A2601605源程序\\', 'PY / R', '三问核心模型、求解、检验与绘图程序，共30个文件（20个PY、10个R）'],
        ['2', 'A2601605数据_不含赛题原始数据\\', 'CSV / JSON', '字段映射、冻结划分、候选字典、数据契约与预处理记录，共18个文件'],
        ['3', 'A2601605结果\\', 'CSV / GZ / NPZ / JSON', '三问正式结果、模型检验与绘图数据，共111个文件'],
        ['4', 'A2601605论文图表\\', 'PNG', '正文图1-18及附图S1-S3，共21张图片'],
        ['5', 'A2601605软件与运行说明.txt', 'TXT', '软件版本、运行命令及Python、R环境快照'],
    ]
    program_rows = [['序号', '支撑材料相对路径', '所属问题']]
    for index, path in enumerate(source_paths, 1):
        question = {'Q1': '问题一', 'Q2': '问题二', 'Q3': '问题三'}.get(path.stem.split('_')[0], '公共绘图')
        program_rows.append([str(index), str(path.relative_to(ATTACHMENT)).replace('/', '\\'), question])
    program_rows.append(['31', txt.name, '全文'])
    spec = {
        'archive_name': 'A2601605附件.zip',
        'intro': '支撑材料压缩包命名为“A2601605附件.zip”。表A.1列出文件组成，表A.2列出30个核心程序及运行说明；所列路径均相对于“A2601605附件”文件夹。',
        'overview': overview,
        'programs': program_rows,
    }
    (AUDIT / 'appendix_a_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, table in [('表A.1_支撑材料文件列表.csv', overview), ('表A.2_程序文件对照.csv', program_rows)]:
        with (AUDIT / name).open('w', encoding='utf-8-sig', newline='') as fh:
            csv.writer(fh).writerows(table)
    inventory = []
    for path in sorted(ATTACHMENT.rglob('*')):
        if path.is_file():
            inventory.append({'relative_path': path.relative_to(ATTACHMENT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes()).hexdigest()})
    assert len(inventory) == 181
    record = {
        'root_name': ATTACHMENT.name,
        'archive_name': 'A2601605附件.zip',
        'files': inventory,
        'counts': expected,
        'root_text_files': [txt.name],
        'file_total': len(inventory),
        'note': '只读核对实际压缩源目录；未创建ZIP，未修改D盘文件。',
        'observations': [
            '实际目录内未见独立AI使用详情PDF或结果工作簿，未在附录中虚列。',
            '实际目录名称已改为A2601605源程序等；运行说明和代码仍有01_源程序、03_结果等旧相对路径，本次附录编辑未修改附件程序。',
        ],
    }
    (AUDIT / '实际附件目录盘点.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(inventory), 'overview_rows': len(overview)-1, 'program_rows': len(program_rows)-1, 'spec': str(AUDIT/'appendix_a_spec.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
