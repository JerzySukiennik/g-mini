#!/usr/bin/env python3
"""Naucz G-Mini odpowiadac ZDANIEM, a nie wycinkiem tekstu.

To jest lekarstwo na konkretna, zmierzona wade [[g-micro]]: po rundach SFT
opartych na PoQuAD model odpowiadal na "Co to jest fotosynteza?" slowem
"koenzym" — poprawnym wycinkiem zrodla i bezuzyteczna odpowiedzia. Przyczyna
nie byla w modelu, tylko w danych: PoQuAD uczy wskazywac fragment, nie mowic.

Sprawdzone przed napisaniem tego pliku (dlatego NIE dokladamy tu PolQA, mimo
ze wygladal obiecujaco):

    zbior          mediana odpowiedzi   jednowyrazowe   pelne zdania
    PolQA          8 znakow / 1 slowo   55%             0%
    PLLuM-Align    46 slow              0.2%            97%

PolQA jest jeszcze bardziej urwany niz PoQuAD — dolozenie go POGLEBILOBY wade,
ktora probujemy usunac. PLLuM-Align jest jej dokladnym przeciwienstwem.

Zrodlo: NASK-PIB/PLLuM-Align, CC-BY-SA-4.0 (NASK Panstwowy Instytut Badawczy) —
pierwszy polski zbior preferencyjny do uczenia asystentow. Bierzemy wylacznie
galaz `chosen`, czyli odpowiedzi ocenione przez ludzi jako lepsze.

Maly (873 unikalne pytania), i to jest w porzadku: przy 178M parametrach liczy
sie spojnosc formy, nie objetosc. Tozsamosc G-Micro nauczyla sie ze 100%
skutecznoscia z kilkudziesieciu przykladow powtorzonych kilka razy.

Wyjscie: data/style_sft.jsonl w tym samym formacie co home_sft.jsonl
({instruction, output, repeats}), wiec build_sft.py wchlania go tak samo.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "NASK-PIB/PLLuM-Align"
FILE = "dialogs.jsonl"

# Gorna granica dlugosci odpowiedzi. 13% odpowiedzi w zrodle przekracza 150
# slow, a model o 178M parametrach i oknie 1024 tokenow nie utrzyma spojnosci
# na takim dystansie — uczenie go tego formatu produkuje dlugie, rozjezdzajace
# sie wypowiedzi. Dolna granica odsiewa resztki wycinkow.
MIN_WORDS = 5
MAX_WORDS = 120

# Maly zbior wysokiej jakosci powtarzamy, tak jak tozsamosc i komendy domowe.
REPEATS = 3


def read_token(env=Path(".env")):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def pairs_from(rows):
    """Pary user -> assistant z galezi `chosen`, bez duplikatow pytania.

    Jedno pytanie wystepuje w wielu wierszach (rozne pary preferencji), wiec
    bez deduplikacji ten sam przyklad wszedlby do treningu kilkanascie razy i
    przewazyl reszte zbioru.
    """
    seen = {}
    for r in rows:
        chosen = r.get("chosen") or []
        for a, b in zip(chosen, chosen[1:]):
            if a.get("role") != "user" or b.get("role") != "assistant":
                continue
            q = (a.get("content") or "").strip()
            ans = (b.get("content") or "").strip()
            if q and ans and q not in seen:
                seen[q] = ans
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/style_sft.jsonl"))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = ap.parse_args()

    path = hf_hub_download(REPO, FILE, repo_type="dataset", token=read_token())
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    seen = pairs_from(rows)
    print(f"zrodlo: {len(rows):,} wierszy -> {len(seen):,} unikalnych pytan")

    kept, too_short, too_long = [], 0, 0
    for q, a in seen.items():
        n = len(a.split())
        if n < MIN_WORDS:
            too_short += 1
            continue
        if n > args.max_words:
            too_long += 1
            continue
        kept.append({"instruction": q, "output": a, "repeats": args.repeats})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    words = [len(r["output"].split()) for r in kept]
    print(f"zapisano {len(kept):,} przykladow do {args.out}")
    print(f"  odrzucone: {too_short} za krotkie, {too_long} za dlugie (>{args.max_words} slow)")
    print(f"  mediana odpowiedzi: {statistics.median(words):.0f} slow")
    print(f"  po powtorzeniach wejdzie do treningu: {len(kept)*args.repeats:,}")
    print("\nZRODLO WYMAGA UZNANIA AUTORSTWA (CC-BY-SA-4.0):")
    print("  PLLuM-Align, NASK Panstwowy Instytut Badawczy")


if __name__ == "__main__":
    main()
