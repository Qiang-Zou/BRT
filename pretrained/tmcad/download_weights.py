"""Download a release asset and verify its pinned SHA-256 before installing it."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Explicit asset URL; otherwise use release.json')
    args = parser.parse_args()
    release = json.loads((HERE / 'release.json').read_text())
    target = HERE / 'checkpoints' / release['asset_name']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != release['sha256']:
            raise ValueError('Existing checkpoint has an unexpected SHA-256')
        print('Verified existing checkpoint:', target)
        return
    url = args.url or release['download_url']
    if not url:
        parser.error('Release destination is not configured yet; use --url after publication')
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as out:
            for block in iter(lambda: source.read(1024 * 1024), b''): out.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != release['sha256']:
            raise ValueError('Downloaded asset SHA-256 mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded and verified:', target)


if __name__ == '__main__': main()
