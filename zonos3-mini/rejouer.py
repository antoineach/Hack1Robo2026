#!/usr/bin/env python3
"""Rejoue les requêtes françaises puis enregistre les WAV dans un dossier à part."""
import argparse
import io
import json
import urllib.error
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent

def call(base, path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body else None
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=220) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code} {exc.read().decode(errors="replace")}') from exc

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--api', default='http://127.0.0.1:1919')
    cli.add_argument('--mesures', type=Path, default=HERE / 'mesures.json')
    cli.add_argument('--sortie', type=Path, default=Path('zonos3-rendus'))
    cli.add_argument('--essai', action='append', help='Identifiant à rejouer, répétable.')
    cli.add_argument('--liste', action='store_true', help='Affiche les identifiants sans appeler le modèle.')
    args = cli.parse_args()
    rows = json.loads(args.mesures.read_text(encoding='utf-8'))
    if args.liste:
        for row in rows:
            print(f"{row['id']}\t{row['family']}\t{row['target']}")
        return
    by_id = {row['id']: row for row in rows}
    unknown = set(args.essai or []) - by_id.keys()
    if unknown:
        cli.error('Essais inconnus ' + ', '.join(sorted(unknown)))
    api = args.api.rstrip('/')
    speakers = json.loads(call(api, '/v2/speakers'))['speakers']
    speaker = next((speaker for speaker in speakers if speaker['label'] == 'Gaspard fr M 1'), None)
    if speaker is None:
        cli.error('La voix Gaspard fr M 1 est absente de /v2/speakers. Vérifie le dossier de voix.')
    args.sortie.mkdir(parents=True, exist_ok=True)
    for name in args.essai or by_id:
        if Path(name).name != name or name in ('', '.', '..'):
            cli.error('Identifiant de fichier invalide ' + repr(name))
        body = {**by_id[name]['request'], 'speaker_embedding_id': speaker['id']}
        raw = call(api, '/v2/generate', body)
        with wave.open(io.BytesIO(raw)) as audio:
            if audio.getnframes() == 0 or audio.getframerate() != 44100:
                raise ValueError(f'WAV inattendu pour {name}')
            duration = audio.getnframes() / audio.getframerate()
        path = args.sortie / f'{name}.wav'
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
        print(f'{path} · {duration:.2f} s', flush=True)

if __name__ == '__main__':
    main()
