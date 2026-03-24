"""
scripts/prepare_fsd50k_tsv.py
==============================
Scarica FSD50K da Zenodo e genera i file TSV per fairseq.

FSD50K è un dataset audio con 51.197 clip etichettate con
l'ontologia AudioSet. È distribuito su Zenodo in 6 zip:
  - dev_audio parte 1-5  (~4.5 GB ciascuno)
  - eval_audio           (~5 GB)

Fairseq si aspetta file TSV nel formato:
  /percorso/cartella
  file1.wav\tn_campioni
  file2.wav\tn_campioni
  ...

Questo script:
  1. Scarica i 6 zip da Zenodo (se non presenti)
  2. Estrae i file audio
  3. Conta i campioni di ogni clip
  4. Genera train.tsv e valid.tsv

USO SU COLAB:
  python scripts/prepare_fsd50k_tsv.py \\
      --output_dir /content/fsd50k_tsv \\
      --audio_dir  /content/FSD50K

USO LOCALE (solo per test):
  python scripts/prepare_fsd50k_tsv.py \\
      --output_dir ./fsd50k_tsv \\
      --audio_dir  ./FSD50K
"""

import os
import csv
import argparse
import subprocess
from pathlib import Path


# ── URL Zenodo FSD50K ─────────────────────────────────────────────────
# DOI: 10.5281/zenodo.4060432
ZENODO_URLS = {
    'dev_1': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01',
    'dev_2': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02',
    'dev_3': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03',
    'dev_4': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04',
    'dev_5': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05',
    'dev_zip': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip',
    'eval_zip': 'https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip',
    'dev_csv': 'https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip',
}

# Frequenza di campionamento target
SAMPLE_RATE = 16000


def scarica_file(url: str, dest: str) -> None:
    """Scarica un file con wget mostrando il progresso."""
    if os.path.exists(dest):
        print(f"  Già presente: {os.path.basename(dest)}")
        return
    print(f"  Scarico: {os.path.basename(dest)} ...")
    subprocess.run(
        ['wget', '-q', '--show-progress', '-O', dest, url],
        check=True
    )
    print(f"  ✅ Scaricato: {os.path.basename(dest)}")


def conta_campioni(wav_path: str) -> int:
    """
    Conta il numero di campioni in un file WAV.

    Usa torchaudio per essere coerente con il resto del progetto.
    Restituisce il numero di campioni DOPO il resample a 16kHz.
    """
    import torchaudio
    import torchaudio.transforms as T

    info = torchaudio.info(wav_path)
    n_campioni_originali = info.num_frames
    sr_originale         = info.sample_rate

    if sr_originale == SAMPLE_RATE:
        return n_campioni_originali

    # Calcola n_campioni dopo resample
    n_campioni_resampled = int(
        n_campioni_originali * SAMPLE_RATE / sr_originale
    )
    return n_campioni_resampled


def genera_tsv(
        audio_dir: str,
        output_path: str,
        max_durata_sec: float = 30.0,
        min_durata_sec: float = 0.5,
) -> int:
    """
    Genera un file TSV da una cartella di file WAV.

    Scorre tutti i .wav nella cartella, conta i campioni,
    e filtra le clip troppo corte o troppo lunghe.

    Args:
        audio_dir:      Cartella con i file .wav
        output_path:    Percorso del TSV da generare
        max_durata_sec: Filtra clip più lunghe di N secondi
        min_durata_sec: Filtra clip più corte di N secondi

    Returns:
        Numero di clip incluse nel TSV
    """
    import torchaudio

    audio_dir = os.path.abspath(audio_dir)
    file_wav  = sorted([
        f for f in os.listdir(audio_dir)
        if f.endswith('.wav') or f.endswith('.flac')
    ])

    if not file_wav:
        print(f"  ⚠️  Nessun file audio in: {audio_dir}")
        return 0

    max_campioni = int(max_durata_sec * SAMPLE_RATE)
    min_campioni = int(min_durata_sec * SAMPLE_RATE)

    incluse  = 0
    escluse  = 0
    errori   = 0
    righe    = []

    print(f"  Analizzo {len(file_wav)} file audio...")

    for i, nome_file in enumerate(file_wav):

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(file_wav)}...")

        percorso = os.path.join(audio_dir, nome_file)

        try:
            n = conta_campioni(percorso)

            if n < min_campioni:
                escluse += 1
                continue
            if n > max_campioni:
                # Tronca invece di escludere — fairseq gestisce il troncamento
                n = max_campioni

            righe.append(f"{nome_file}\t{n}")
            incluse += 1

        except Exception as e:
            errori += 1
            if errori <= 5:
                print(f"  ⚠️  Errore su {nome_file}: {e}")

    # Scrivi il TSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(audio_dir + '\n')
        f.write('\n'.join(righe) + '\n')

    print(f"  ✅ TSV generato: {output_path}")
    print(f"     Incluse:  {incluse}")
    print(f"     Escluse:  {escluse}  (troppo corte)")
    print(f"     Errori:   {errori}")

    return incluse


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.audio_dir,  exist_ok=True)

    dev_dir  = os.path.join(args.audio_dir, 'dev_audio')
    eval_dir = os.path.join(args.audio_dir, 'eval_audio')

    # ── Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n[1/4] Download FSD50K da Zenodo...")
        download_dir = os.path.join(args.audio_dir, 'zips')
        os.makedirs(download_dir, exist_ok=True)

        for nome, url in ZENODO_URLS.items():
            dest = os.path.join(download_dir, os.path.basename(url))
            scarica_file(url, dest)
    else:
        print("\n[1/4] Download saltato (--skip_download)")
        download_dir = os.path.join(args.audio_dir, 'zips')

    # ── Estrazione ────────────────────────────────────────────────────
    if not args.skip_extract:
        print("\n[2/4] Estrazione zip...")

        # Dev audio: è uno zip multi-volume (.z01-.z05 + .zip)
        dev_zip = os.path.join(download_dir, 'FSD50K.dev_audio.zip')
        if not os.path.exists(dev_dir) or len(os.listdir(dev_dir)) < 1000:
            print("  Estraggo dev_audio (zip multi-volume)...")
            os.makedirs(dev_dir, exist_ok=True)
            subprocess.run(
                ['unzip', '-q', dev_zip, '-d', args.audio_dir],
                check=True
            )
            print("  ✅ dev_audio estratto")
        else:
            print(f"  dev_audio già estratto ({len(os.listdir(dev_dir))} file)")

        # Eval audio
        eval_zip = os.path.join(download_dir, 'FSD50K.eval_audio.zip')
        if not os.path.exists(eval_dir) or len(os.listdir(eval_dir)) < 100:
            print("  Estraggo eval_audio...")
            os.makedirs(eval_dir, exist_ok=True)
            subprocess.run(
                ['unzip', '-q', eval_zip, '-d', args.audio_dir],
                check=True
            )
            print("  ✅ eval_audio estratto")
        else:
            print(f"  eval_audio già estratto ({len(os.listdir(eval_dir))} file)")

        # Ground truth CSV
        gt_zip = os.path.join(download_dir, 'FSD50K.ground_truth.zip')
        gt_dir = os.path.join(args.audio_dir, 'ground_truth')
        if not os.path.exists(gt_dir):
            subprocess.run(
                ['unzip', '-q', gt_zip, '-d', args.audio_dir],
                check=True
            )
            print("  ✅ ground_truth estratto")

    else:
        print("\n[2/4] Estrazione saltata (--skip_extract)")

    # ── Verifica ──────────────────────────────────────────────────────
    print("\n[3/4] Verifica file audio...")

    n_dev  = len([f for f in os.listdir(dev_dir)
                  if f.endswith('.wav')]) if os.path.exists(dev_dir) else 0
    n_eval = len([f for f in os.listdir(eval_dir)
                  if f.endswith('.wav')]) if os.path.exists(eval_dir) else 0

    print(f"  dev_audio:  {n_dev} file wav  (atteso: ~40.966)")
    print(f"  eval_audio: {n_eval} file wav  (atteso: ~10.231)")

    if n_dev < 1000:
        print("  ❌ dev_audio sembra vuoto — controlla l'estrazione")
        return

    # ── Genera TSV ────────────────────────────────────────────────────
    print("\n[4/4] Generazione TSV per fairseq...")

    # train.tsv ← dev_audio (40k clip, usato per pretraining)
    train_tsv = os.path.join(args.output_dir, 'train.tsv')
    print("\n  train.tsv (dev_audio):")
    n_train = genera_tsv(
        audio_dir   = dev_dir,
        output_path = train_tsv,
    )

    # valid.tsv ← eval_audio (10k clip, usato come validation)
    valid_tsv = os.path.join(args.output_dir, 'valid.tsv')
    print("\n  valid.tsv (eval_audio):")
    n_valid = genera_tsv(
        audio_dir   = eval_dir,
        output_path = valid_tsv,
    )

    # ── Riepilogo ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  TSV PRONTI PER FAIRSEQ")
    print(f"{'='*55}")
    print(f"  train.tsv:  {n_train} clip → {train_tsv}")
    print(f"  valid.tsv:  {n_valid} clip → {valid_tsv}")
    print(f"\n  Prossimo step: copia i TSV su Google Drive")
    print(f"  e usali nel notebook PartA_pretrain.ipynb")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara FSD50K per il pretraining MAE-AST con fairseq"
    )
    parser.add_argument(
        '--output_dir',
        default='/content/fsd50k_tsv',
        help='Cartella dove salvare i file TSV'
    )
    parser.add_argument(
        '--audio_dir',
        default='/content/FSD50K',
        help='Cartella dove scaricare/cercare FSD50K'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Salta il download (usa file già presenti)'
    )
    parser.add_argument(
        '--skip_extract',
        action='store_true',
        help='Salta la decompressione (usa file già estratti)'
    )

    args = parser.parse_args()
    main(args)