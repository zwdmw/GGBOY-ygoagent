"""Materialize a hash-pinned historical pool without changing the source weights."""
import argparse
import json
import os
import shutil
from pathlib import Path

from ygo_sky.paths import contained, project_root, read_json, write_json
from ygo_sky.resources import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = project_root()
manifest = read_json(args.manifest)
if manifest.get('schema') != 'ygo-sky-opponent-pool/v1':
    raise ValueError('Unsupported pool manifest')
verified = []
for row in manifest['members']:
    source = contained(root, row['path'])
    metadata = read_json(Path(str(source) + '.json'))
    if (sha256(source) != row['sha256'] or metadata['sha256'] != row['sha256']
            or metadata['global_step'] != row['step']):
        raise ValueError('Historical pool identity mismatch: ' + row['path'])
    verified.append(source)
args.output.mkdir(parents=True, exist_ok=False)
for source in verified:
    for file in (source, Path(str(source) + '.json')):
        target = args.output / file.name
        try:
            os.link(file, target)
        except OSError:
            shutil.copy2(file, target)
write_json(args.output / 'pool.json', manifest)
print(json.dumps({'pool': str(args.output.resolve()), 'members': len(verified)}))
