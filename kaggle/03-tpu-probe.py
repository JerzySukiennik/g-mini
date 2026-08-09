"""Kaggle kernel: czy TPU v3-8 oplaca sie dla G-Mini — pomiar, nie ekstrapolacja.

Ten projekt trzy razy przejechal sie na przewidywaniu przepustowosci z papieru
(raz zakladajac stala wydajnosc, raz sonda robiaca krok optymalizatora po kazdym
micro-batchu, raz mierzac ze zlym slownikiem). Papier mowi, ze v3-8 ma ~420
TFLOPS bf16 wobec ~130 dla T4x2, czyli ~3x. Ta sonda ma sprawdzic, ile z tego
zostaje naprawde.

Zaprojektowana tak, zeby JEDEN przebieg byl informatywny nawet gdy sie wywroci:
kazda faza wypisuje wynik zanim zacznie sie nastepna, a porazka pozniejszej nie
kasuje wiedzy z wczesniejszej.

  faza 0  srodowisko: czy torch_xla w ogole jest, jaka wersja, ile rdzeni
  faza 1  jeden rdzen: model sie buduje, krok przechodzi, ile tok/s
  faza 2  osiem rdzeni przez xmp.spawn: laczna przepustowosc

Dane sa LOSOWE. Mierzymy predkosc, nie ucze modelu, wiec podpinanie prawdziwego
korpusu tylko dodaloby tryb awarii (mount, sciezki) bez zadnej wartosci.

Punkt odniesienia z realnego treningu na T4x2: **10 100 tok/s** przy 65 536
tokenach na krok (micro-batch 4, akumulacja 16, kontekst 1024).
"""

import os
import subprocess
import sys
import time

REPO = "https://github.com/JerzySukiennik/g-mini.git"
WORK = "/kaggle/working"

T4_BASELINE = 10_100          # zmierzone tok/s na sesji 1 i 2
BLOCK = 1024
STEPS_MEASURED = 30           # po rozgrzewce; XLA kompiluje przy pierwszych krokach
WARMUP = 10

if os.path.exists(f"{WORK}/g-mini"):
    subprocess.run(["git", "-C", f"{WORK}/g-mini", "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-mini"], check=True)
os.chdir(f"{WORK}/g-mini")
sys.path.insert(0, f"{WORK}/g-mini")

print("=" * 60)
print("FAZA 0 — srodowisko")
print("=" * 60, flush=True)

import torch
print(f"torch {torch.__version__}")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"torch_xla {getattr(torch_xla, '__version__', '?')}")
except ImportError as e:
    print(f"BRAK torch_xla: {e}")
    print("\nWNIOSEK: srodowisko Kaggle TPU nie ma preinstalowanego PyTorch/XLA.")
    print("Port wymagalby instalacji w kernelu (wolne, wersje musza pasowac do")
    print("wersji runtime TPU) albo przepisania treningu na JAX/TF.")
    sys.exit(0)

try:
    devices = xm.get_xla_supported_devices()
    print(f"rdzenie XLA widoczne: {len(devices)}")
    print(f"lista: {devices}")
except Exception as e:
    print(f"nie moge wylistowac urzadzen: {e}")

print("\n" + "=" * 60)
print("FAZA 1 — jeden rdzen")
print("=" * 60, flush=True)

from model.gpt import GPT, GPTConfig


def build():
    cfg = GPTConfig()
    model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def measure(model, device, batch, label):
    """Zwraca tok/s albo None. Krok jest pelny: forward, backward, optymalizator —
    pomiar samego forwardu zawyzalby wynik kilkukrotnie."""
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab = model.config.vocab_size if hasattr(model, "config") else 48000

    def one_step():
        x = torch.randint(0, vocab, (batch, BLOCK), device=device)
        y = torch.randint(0, vocab, (batch, BLOCK), device=device)
        _, loss = model(x, targets=y, return_logits=False)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        xm.mark_step()          # bez tego XLA buduje graf w nieskonczonosc

    try:
        for _ in range(WARMUP):
            one_step()
        xm.wait_device_ops()
        t0 = time.time()
        for _ in range(STEPS_MEASURED):
            one_step()
        xm.wait_device_ops()
        dt = time.time() - t0
    except Exception as e:
        print(f"{label}: BLAD — {type(e).__name__}: {str(e)[:300]}")
        return None

    tps = batch * BLOCK * STEPS_MEASURED / dt
    print(f"{label}: batch {batch} -> {tps:,.0f} tok/s ({dt/STEPS_MEASURED:.3f} s/krok)")
    return tps


single = None
try:
    dev = xm.xla_device()
    print(f"urzadzenie: {dev}")
    model, cfg, n = build()
    print(f"parametry: {n:,}")
    # Rosnacy batch: 128 GB HBM na v3-8 to duzo wiecej niz 16 GB T4, wiec
    # ograniczenie, ktore zmusilo nas do micro-batcha 4, moze tu nie istniec.
    for b in (4, 8, 16):
        r = measure(model, dev, b, f"1 rdzen, batch {b}")
        if r is None:
            break
        single = max(single or 0, r)
except Exception as e:
    print(f"faza 1 przerwana: {type(e).__name__}: {str(e)[:300]}")

if single:
    print(f"\n1 rdzen daje {single:,.0f} tok/s wobec {T4_BASELINE:,} na calym T4x2 "
          f"({single/T4_BASELINE:.2f}x)")

print("\n" + "=" * 60)
print("FAZA 2 — osiem rdzeni")
print("=" * 60, flush=True)

try:
    import torch_xla.distributed.xla_multiprocessing as xmp

    def _worker(rank):
        d = xm.xla_device()
        m, _, _ = build()
        r = measure(m, d, 8, f"rdzen {rank}")
        if rank == 0 and r:
            print(f"\nSZACUNEK LACZNIE: ~{r*8:,.0f} tok/s "
                  f"({r*8/T4_BASELINE:.1f}x wobec T4x2)")

    xmp.spawn(_worker, args=(), nprocs=8, start_method="fork")
except Exception as e:
    print(f"faza 2 nie przeszla: {type(e).__name__}: {str(e)[:400]}")
    if single:
        print(f"\nMozna zgadywac ~{single*8:,.0f} tok/s przy 8 rdzeniach, ale to "
              f"JEST ekstrapolacja — dokladnie ten blad, ktory ten projekt "
              f"popelnil juz trzy razy. Traktowac jako gorna granice.")

print("\nsonda zakonczona", flush=True)
