"""
scripts/prepare_fsd50k_tsv.py
==============================
Scarica FSD50K da Zenodo e genera i file TSV per il pretraining MAE-AST.

FORMATO TSV RICHIESTO:
    /
    /percorso/assoluto/file1.flac\tn_campioni_audio
    /percorso/assoluto/file2.flac\tn_campioni_audio

Note importanti:
- Prima riga: "/" (solo uno slash)
- Percorsi assoluti completi al file audio
- I file audio sono .flac a 44.100 Hz (frequenza nativa di FSD50K)
- Il dataloader fa il resample a 16kHz internamente
- Il numero di campioni nel TSV serve solo per ordinare i batch

USO SU COLAB:
  python scripts/prepare_fsd50k_tsv.py \
      --output_dir /content/fsd50k_tsv \
      --audio_dir  /content/FSD50K
"""

import os
import argparse
import subprocess
import random


# Soglie in campioni a 44.100 Hz (frequenza nativa di FSD50K)
SAMPLE_RATE     = 44100
MIN_SAMPLES     = 44100    # 1 secondo
MAX_SAMPLES     = 705600   # 16 secondi

# Split train/valid
TRAIN_RATIO = 0.9

ZENODO_URLS = {
    'dev_1':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01',
    'dev_2':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02',
    'dev_3':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03',
    'dev_4':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04',
    'dev_5':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05',
    'dev_zip': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip',
}


def scarica_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  Gia presente: {os.path.basename(dest)}")
        return
    print(f"  Scarico: {os.path.basename(dest)} ...")
    subprocess.run(['wget', '-q', '--show-progress', '-O', dest, url], check=True)
    print(f"  OK: {os.path.basename(dest)}")


def converti_wav_in_flac(audio_dir: str) -> int:
    """
    Converte tutti i .wav in .flac mantenendo il sample rate nativo (44.1kHz).
    Il dataloader fa il resample a 16kHz internamente.
    """
    file_wav  = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    file_flac = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]

    if not file_wav:
        print(f"  Nessun .wav trovato (forse gia convertiti: {len(file_flac)} .flac)")
        return len(file_flac)

    if len(file_flac) >= len(file_wav):
        print(f"  Conversione gia fatta: {len(file_flac)} .flac presenti")
        return len(file_flac)

    print(f"  Converto {len(file_wav)} .wav -> .flac (44.1kHz nativo)...")

    convertiti = 0
    errori     = 0

    for i, nome_wav in enumerate(sorted(file_wav)):
        percorso_wav  = os.path.join(audio_dir, nome_wav)
        percorso_flac = os.path.join(audio_dir, nome_wav.replace('.wav', '.flac'))

        if os.path.exists(percorso_flac):
            convertiti += 1
            os.remove(percorso_wav)
            continue

        try:
            subprocess.run([
                'ffmpeg', '-i', percorso_wav,
                '-c:a', 'flac',
                '-ac', '1',
                percorso_flac,
                '-y', '-loglevel', 'error'
            ], check=True)
            os.remove(percorso_wav)
            convertiti += 1
        except Exception as e:
            errori += 1
            if errori <= 5:
                print(f"  Errore su {nome_wav}: {e}")

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(file_wav)} ...")

    print(f"  Completato: {convertiti} .flac, {errori} errori")
    return convertiti


def conta_campioni_flac(percorso: str) -> int:
    """Conta i campioni audio in un file .flac usando torchaudio.info()."""
    import torchaudio
    info = torchaudio.info(percorso)
    return info.num_frames


def genera_tsv_split(
    audio_dir: str,
    train_path: str,
    valid_path: str,
    seed: int = 42,
) -> tuple:
    """
    Genera train.tsv e valid.tsv da una singola cartella audio (dev_audio).
    Split 90% train, 10% valid. Solo file tra MIN_SAMPLES e MAX_SAMPLES.
    """
    audio_dir = os.path.abspath(audio_dir)
    file_flac = sorted([f for f in os.listdir(audio_dir) if f.endswith('.flac')])

    if not file_flac:
        print(f"  Nessun .flac in: {audio_dir}")
        return 0, 0

    print(f"  Analizzo {len(file_flac)} file .flac...")

    righe    = []
    escluse  = 0
    troncate = 0
    errori   = 0

    for i, nome in enumerate(file_flac):
        percorso = os.path.join(audio_dir, nome)
        try:
            n = conta_campioni_flac(percorso)

            if n < MIN_SAMPLES:
                escluse += 1
                continue

            if n > MAX_SAMPLES:
                n = MAX_SAMPLES
                troncate += 1

            righe.append(f"{percorso}\t{n}")

        except Exception as e:
            errori += 1
            if errori <= 5:
                print(f"  Errore su {nome}: {e}")

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(file_flac)} ...")

    print(f"  Clip valide: {len(righe)}")
    print(f"  Escluse:     {escluse}  (sotto {MIN_SAMPLES} campioni = {MIN_SAMPLES/SAMPLE_RATE:.0f}s)")
    print(f"  Troncate:    {troncate} (sopra {MAX_SAMPLES} campioni = {MAX_SAMPLES/SAMPLE_RATE:.0f}s)")
    print(f"  Errori:      {errori}")

    # Shuffle e split 90/10
    rng = random.Random(seed)
    rng.shuffle(righe)

    n_train = int(len(righe) * TRAIN_RATIO)
    righe_train = righe[:n_train]
    righe_valid = righe[n_train:]

    # Scrivi train.tsv
    os.makedirs(os.path.dirname(os.path.abspath(train_path)), exist_ok=True)
    with open(train_path, 'w') as f:
        f.write('/\n')
        f.write('\n'.join(righe_train) + '\n')

    # Scrivi valid.tsv
    with open(valid_path, 'w') as f:
        f.write('/\n')
        f.write('\n'.join(righe_valid) + '\n')

    print(f"\n  train.tsv: {len(righe_train)} clip")
    print(f"  valid.tsv: {len(righe_valid)} clip")

    return len(righe_train), len(righe_valid)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.audio_dir,  exist_ok=True)

    dev_dir = os.path.join(args.audio_dir, 'FSD50K.dev_audio')

    # Fase 1: Download (solo dev_audio)
    if not args.skip_download:
        print("\n[1/4] Download FSD50K dev_audio da Zenodo (~22 GB)...")
        dl_dir = os.path.join(args.audio_dir, 'zips')
        os.makedirs(dl_dir, exist_ok=True)
        for nome, url in ZENODO_URLS.items():
            scarica_file(url, os.path.join(dl_dir, os.path.basename(url)))
    else:
        print("\n[1/4] Download saltato")
        dl_dir = os.path.join(args.audio_dir, 'zips')

    # Fase 2: Estrazione con 7z (gli zip di FSD50K sono multi-volume)
    if not args.skip_extract:
        print("\n[2/4] Estrazione con 7z (zip multi-volume)...")
        n_dev = len([f for f in os.listdir(dev_dir)
                     if f.endswith(('.wav', '.flac'))]) if os.path.exists(dev_dir) else 0
        if n_dev < 1000:
            dev_zip = os.path.join(dl_dir, 'FSD50K.dev_audio.zip')
            print("  Estraggo FSD50K.dev_audio...")
            subprocess.run(['7z', 'x', dev_zip, f'-o{args.audio_dir}', '-y'],
                           check=True, stdout=subprocess.DEVNULL)
            print("  OK dev_audio")
        else:
            print(f"  dev_audio gia estratto ({n_dev} file)")

        # Elimina zip per liberare spazio
        import shutil
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
            print("  Zip eliminati")
    else:
        print("\n[2/4] Estrazione saltata")

    # Fase 3: Conversione wav -> flac (mantenendo 44.1kHz nativo)
    if not args.skip_convert:
        print("\n[3/4] Conversione .wav -> .flac (44.1kHz mono)...")
        converti_wav_in_flac(dev_dir)
    else:
        print("\n[3/4] Conversione saltata")

    # Fase 4: Genera TSV con split 90/10
    print("\n[4/4] Generazione TSV (split 90/10 da dev_audio)...")
    n_dev = len([f for f in os.listdir(dev_dir) if f.endswith('.flac')]) if os.path.exists(dev_dir) else 0
    print(f"  dev_audio: {n_dev} .flac  (atteso: ~40.966)")
    if n_dev < 1000:
        print("  ERRORE: dev_audio sembra vuoto")
        return

    train_tsv = os.path.join(args.output_dir, 'train.tsv')
    valid_tsv = os.path.join(args.output_dir, 'valid.tsv')

    n_train, n_valid = genera_tsv_split(dev_dir, train_tsv, valid_tsv)

    # Verifica formato
    with open(train_tsv) as f:
        prima = f.readline().strip()
        seconda = f.readline().strip()
    print(f"\n  Prima riga:  '{prima}'  (atteso: '/')")
    print(f"  Esempio:     '{seconda[:80]}'")
    assert prima == '/', f"ERRORE formato TSV: prima riga deve essere '/', trovato '{prima}'"

    print(f"\n{'='*55}")
    print(f"  TSV PRONTI")
    print(f"{'='*55}")
    print(f"  train.tsv: {n_train} clip (90%)")
    print(f"  valid.tsv: {n_valid} clip (10%)")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/content/fsd50k_tsv')
    parser.add_argument('--audio_dir',  default='/content/FSD50K')
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--skip_extract',  action='store_true')
    parser.add_argument('--skip_convert',  action='store_true')
    args = parser.parse_args()
    main(args)
