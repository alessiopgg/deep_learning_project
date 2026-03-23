"""
scripts/inspect_checkpoint.py
==============================
Ispeziona un checkpoint MAE-AST in formato fairseq.

Carica il file .pt e stampa la struttura interna:
  - Chiavi di primo livello del dizionario
  - Lista completa dei tensori con nome, shape e n. parametri
  - Sezioni rilevanti (encoder, decoder, ecc.)

NON modifica nulla — è solo lettura.
Eseguire PRIMA di convert_checkpoint.py per capire
la struttura del checkpoint prima di convertirlo.

Uso:
    python scripts/inspect_checkpoint.py <percorso_checkpoint.pt>

Esempio:
    python scripts/inspect_checkpoint.py chunk_patch_75_12LayerEncoder.pt
"""

import sys
import os
import torch


def ispeziona(percorso: str) -> None:

    # ── Verifica esistenza file ───────────────────────────────────────
    if not os.path.exists(percorso):
        print(f"\n❌  File non trovato: {percorso}")
        print("\n    Scaricalo con:")
        print("    (Patch) wget https://www.cs.utexas.edu/~harwath/"
              "model_checkpoints/mae_ast/chunk_patch_75_12LayerEncoder.pt")
        print("    (Frame) wget https://www.cs.utexas.edu/~harwath/"
              "model_checkpoints/mae_ast/random_frame_75_12LayerEncoder.pt")
        sys.exit(1)

    dim_mb = os.path.getsize(percorso) / 1e6
    print(f"\n{'='*60}")
    print(f"  ISPEZIONE CHECKPOINT")
    print(f"  File:       {os.path.basename(percorso)}")
    print(f"  Dimensione: {dim_mb:.1f} MB")
    print(f"{'='*60}")

    # ── Caricamento ───────────────────────────────────────────────────
    # map_location='cpu': carica in RAM, non serve GPU
    print("\n  Caricamento in corso...")
    ck = torch.load(percorso, map_location="cpu")
    print("  ✅ Caricato\n")

    # ── Struttura di primo livello ────────────────────────────────────
    print(f"{'='*60}")
    print("  CHIAVI DI PRIMO LIVELLO")
    print(f"{'='*60}")
    for k, v in ck.items():
        if isinstance(v, dict):
            desc = f"dict con {len(v)} chiavi"
        elif isinstance(v, torch.Tensor):
            desc = f"Tensor {list(v.shape)}"
        else:
            desc = str(type(v).__name__)
        print(f"  '{k}': {desc}")

    # ── Estrai i pesi del modello ─────────────────────────────────────
    if "model" in ck:
        state = ck["model"]
        print(f"\n  → Pesi trovati in: ck['model']")
    elif "state_dict" in ck:
        state = ck["state_dict"]
        print(f"\n  → Pesi trovati in: ck['state_dict']")
    elif any(isinstance(v, torch.Tensor) for v in ck.values()):
        state = ck
        print(f"\n  → Il checkpoint stesso è il dizionario pesi")
    else:
        print("\n  ❌ Struttura non riconosciuta.")
        sys.exit(1)

    # ── Statistiche ───────────────────────────────────────────────────
    n_tensori   = sum(1 for v in state.values() if isinstance(v, torch.Tensor))
    n_parametri = sum(
        v.numel() for v in state.values() if isinstance(v, torch.Tensor)
    )

    print(f"\n{'='*60}")
    print(f"  PESI DEL MODELLO")
    print(f"  Tensori totali:   {n_tensori}")
    print(f"  Parametri totali: {n_parametri:,}  (~{n_parametri/1e6:.1f}M)")
    print(f"{'='*60}")

    # ── Sezioni (prefissi) ────────────────────────────────────────────
    sezioni: dict = {}
    for k in state.keys():
        prefisso = k.split(".")[0]
        sezioni[prefisso] = sezioni.get(prefisso, 0) + 1

    print("\n  SEZIONI (prefissi delle chiavi):")
    for sez, n in sorted(sezioni.items()):
        print(f"    '{sez}.*'  →  {n} tensori")

    # ── Lista completa ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LISTA COMPLETA CHIAVI")
    print(f"{'='*60}")
    fmt = "  {:<65} {:<22} {:>12}"
    print(fmt.format("Chiave", "Shape", "Parametri"))
    print("  " + "-" * 100)

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            shape_str  = str(list(v.shape))
            n_par      = v.numel()
            print(fmt.format(k[:65], shape_str[:22], f"{n_par:,}"))
        else:
            print(f"  {k:<65} (non-Tensor: {type(v).__name__})")

    # ── Salva lista su file ───────────────────────────────────────────
    out_file = os.path.splitext(os.path.basename(percorso))[0] + "_chiavi.txt"
    with open(out_file, "w") as f:
        f.write(f"Checkpoint: {percorso}\n")
        f.write(f"Parametri totali: {n_parametri:,}\n\n")
        f.write("CHIAVI:\n")
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                f.write(f"{k}  {list(v.shape)}\n")

    print(f"\n  📄 Lista salvata in: {out_file}")

    # ── Suggerimento prossimo step ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PROSSIMO STEP")
    print(f"{'='*60}")
    print("\n  Esegui la conversione:")
    print(f"  python scripts/convert_checkpoint.py "
          f"{percorso} <output.pth>")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/inspect_checkpoint.py <checkpoint.pt>")
        print()
        print("Esempio:")
        print("  python scripts/inspect_checkpoint.py "
              "chunk_patch_75_12LayerEncoder.pt")
        sys.exit(1)

    ispeziona(sys.argv[1])