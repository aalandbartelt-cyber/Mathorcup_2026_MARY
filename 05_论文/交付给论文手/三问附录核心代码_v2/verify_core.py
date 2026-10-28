"""Static/hash/import checks only; never call a modelling function."""
from pathlib import Path
from hashlib import sha256
import ast
import importlib.util
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    digest, relative = line.split('  ', 1)
    assert sha256((root / relative).read_bytes()).hexdigest() == digest, relative
modules = []
for path in sorted(root.glob('Q*/*.py')):
    source = path.read_text(encoding='utf-8')
    ast.parse(source)
    compile(source, path.name, 'exec')
    sys.path.insert(0, str(path.parent))
    name = 'appendix_' + path.parent.name[:2] + '_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    modules.append(path.relative_to(root).as_posix())
print('SHA256 verified; syntax and imports valid for', len(modules), 'core modules.')
print('No training, sampling, ranking or result regeneration was executed.')
