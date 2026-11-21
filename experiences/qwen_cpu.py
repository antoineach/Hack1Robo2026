"""Génère un récit Qwen et conserve les positions nécessaires à la synthèse vocale."""
import argparse
from dataclasses import dataclass
import json
from os.path import commonprefix
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hack1robo import GemmaDecoder, Profile

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
AMORCE = "J’ai poussé la porte du cinéma. L’eau me montait aux chevilles et la salle sentait la poussière mouillée."
PROMPT = "Poursuis cette scène fictive en français à la première personne, au passé. Trois courts paragraphes."


@dataclass(frozen=True)
class Steps(Profile):
    interval: int = 12

    def at(self, t):
        if t < self.onset:
            return self.nominal, 0.0
        step = (t - self.onset) // self.interval
        tenths = 9 + step if step <= 5 else max(1, 19 - step)
        return tenths / 10, 0.0


@dataclass(frozen=True)
class DeepCrash(Profile):
    lower_floor: float = 0.15
    floor_at: int = 105

    def at(self, t):
        if t <= self.crash_at:
            return super().at(t)
        if t >= self.floor_at:
            return self.lower_floor, 0.0
        progress = (t - self.crash_at) / (self.floor_at - self.crash_at)
        return self.crash + progress * (self.lower_floor - self.crash), 0.0


class PrefixProcessor:
    def __init__(self, tokenizer, prefix):
        self.tokenizer = tokenizer
        self.prefix = prefix

    def apply_chat_template(self, messages, *, enable_thinking, add_generation_prompt, **kwargs):
        text = "".join(part["text"] for part in messages[0]["content"])
        dialogue = [{"role": "user", "content": text}, {"role": "assistant", "content": self.prefix}]
        return self.tokenizer.apply_chat_template(dialogue, continue_final_message=True, **kwargs)


class QwenDecoder(GemmaDecoder):
    def __init__(self, prefix):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = MODEL
        self.device = torch.device("cpu")
        self.processor = PrefixProcessor(AutoTokenizer.from_pretrained(MODEL, revision=REVISION), prefix)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cpu"},
        ).eval()
        self.model.generation_config.repetition_penalty = 1.0


def with_text_offsets(row, tokenizer):
    text = row["text"]
    ids = [entry["id"] for entry in row["tokens"]]
    if tokenizer.decode(ids, skip_special_tokens=True) != text:
        raise ValueError("Le texte et ses tokens ne correspondent pas.")
    previous = 0
    for i, entry in enumerate(row["tokens"], 1):
        prefix = tokenizer.decode(ids[:i], skip_special_tokens=True)
        end = len(commonprefix((text, prefix)))
        if end < previous:
            raise ValueError("Les positions de caractères ne sont pas monotones.")
        entry["character_end"] = previous = end
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sortie", type=Path, required=True)
    parser.add_argument("--id")
    parser.add_argument("--intervalle", type=int, choices=(6, 12), default=12)
    parser.add_argument("--rechute", type=float, choices=(0.15, 0.10),
                        help="Courbe linéaire prolongée jusqu’à ce palier au token 105.")
    parser.add_argument("--graine", type=int, default=17)
    parser.add_argument("--amorce", default=AMORCE)
    args = parser.parse_args()
    if args.rechute is not None:
        profile, min_tokens, max_tokens = DeepCrash(lower_floor=args.rechute), 112, 144
        default_id = f"cinema-rechute-{round(args.rechute * 100)}"
    else:
        profile = Steps(interval=args.intervalle)
        min_tokens, max_tokens = {6: (143, 160), 12: (248, 276)}[args.intervalle]
        default_id = "cinema-paliers-rapides" if args.intervalle == 6 else "cinema-paliers-lents"
    run_id = args.id or default_id
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id):
        parser.error("--id doit contenir uniquement des minuscules, chiffres et tirets")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    decoder = QwenDecoder(args.amorce)
    run = decoder.generate(PROMPT, profile, seed=args.graine, min_tokens=min_tokens, max_tokens=max_tokens)
    if isinstance(profile, Steps):
        run["profile"].update(peak_at=profile.onset + 5 * profile.interval,
                              crash_at=profile.onset + 18 * profile.interval, crash=0.1)
    run.update(id=run_id, assistant_prefix=args.amorce, prompt=PROMPT)
    with_text_offsets(run, decoder.processor.tokenizer)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(run, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(args.amorce + " " + run["text"].lstrip())


if __name__ == "__main__":
    main()
