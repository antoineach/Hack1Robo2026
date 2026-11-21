"""Prépare deux prises ZONOS3 à partir du texte et de la courbe d'un récit Qwen."""
import argparse
from itertools import groupby
import json
from pathlib import Path


def build_cases(run, seed=42, max_tokens=720, follow="reading"):
    continuation = run["text"]
    prefix = run.get("assistant_prefix") or ""
    header = prefix + (" " if prefix and continuation and not continuation[0].isspace() else "")
    text = header + continuation
    trace = run["tokens"]
    if not continuation.strip() or not trace:
        raise ValueError("La suite générée ou sa trace est vide.")
    if max_tokens <= 0:
        raise ValueError("Le budget ZONOS3 doit être positif.")
    nominal = run["profile"]["nominal"]
    peak = max(entry["temperature"] for entry in trace)
    floor = min(entry["temperature"] for entry in trace)
    def intensity(entry):
        # Un dixième de température par bande. Un contrôle continu restait trop
        # fin pour délimiter des passages vocaux comparables.
        return round(entry["temperature"] * 10)

    spans = []
    cursor = 0
    for key, group in groupby(trace, key=intensity):
        end = cursor
        for entry in group:
            offset = entry["character_end"]
            if type(offset) is not int or not end <= offset <= len(continuation):
                raise ValueError("Position de caractère invalide dans la trace.")
            end = offset
        temperature = key / 10
        controls = {}
        if temperature > nominal and peak > nominal:
            weight = min(1, (temperature - nominal) / (peak - nominal))
            controls = {"emotions.confused": round(2.2 * weight, 3),
                        "disfluency.disfluent": round(3.5 * weight, 3),
                        "expressive.expressive": round(4 * weight, 3)}
        elif temperature < nominal and floor < nominal:
            weight = min(1, (nominal - temperature) / (nominal - floor))
            controls = {"delivery.whisper": round(2.2 * weight, 3),
                        "mood.arousal": round(-2 * weight, 3)}
        if end > cursor and controls:
            spans.append({"start": len(header) + cursor, "end": len(header) + end, "controls": controls})
        cursor = end
    if cursor != len(continuation):
        raise ValueError("La trace ne couvre pas le texte généré.")

    shared = {"text": text, "language": "fr_fr", "speaker_mode": "embedding",
              "seed": seed, "stream": False, "format": "wav", "max_tokens": max_tokens}
    follow_options = {"control_span_follow": follow, "control_span_rate": 1.05}
    return [
        {"id": run["id"] + "-temoin", "family": "Qwen → ZONOS3", "target": "Même texte, voix témoin",
         "request": shared},
        {"id": run["id"] + "-courbe", "family": "Qwen → ZONOS3", "target": "Vertige à la montée et souffle à la rechute",
         "request": {**shared, **follow_options, "control_spans": spans}},
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("--essai", help="Identifiant à choisir si le JSON contient une liste.")
    cli.add_argument("--sortie", type=Path, required=True)
    cli.add_argument("--suivi", choices=("reading", "rate"), default="reading")
    cli.add_argument("--max-tokens", type=int, default=720)
    args = cli.parse_args()
    records = json.loads(args.trace.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    matches = [row for row in records if args.essai is None or row["id"] == args.essai]
    if len(matches) != 1:
        cli.error("Sélectionne exactement un récit avec --essai.")
    result = build_cases(matches[0], max_tokens=args.max_tokens, follow=args.suivi)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.sortie} · {len(result[1]['request']['control_spans'])} passages pilotés")


if __name__ == "__main__":
    main()
