"""
scripts/prepare_fsd50k_tsv.py
==============================
Scarica FSD50K da Zenodo e genera i file TSV per fairseq-hydra-train.

FORMATO TSV RICHIESTO (dal README MAE-AST e dal codice del task):
    /
    /percorso/assoluto/file1.flac\tn_campioni_audio
    /percorso/assoluto/file2.flac\tn_campioni_audio

Note importanti:
- Prima riga: "/" (solo uno slash)
- Percorsi assoluti completi al file audio
- n_campioni = campioni audio GREZZI a 16kHz (non frame fbank)
  Il task mae_ast_pretraining calcola le fbank internamente.
- Il dataloader fairseq accetta .flac e .mkv (non .wav)
  Per questo convertiamo FSD50K da .wav a .flac

PARAMETRI TASK (da config/pretrain/mae_ast.yaml):
  sample_rate: 16000
  max_sample_size: 250000  (~15.6 secondi)
  min_sample_size: 32000   (~2 secondi)

USO SU COLAB:
  python scripts/prepare_fsd50k_tsv.py \
      --output_dir /content/fsd50k_tsv \
      --audio_dir  /content/FSD50K
"""

import os
import argparse
import subprocess


ZENODO_URLS = {
    'dev_1':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01',
    'dev_2':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02',
    'dev_3':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03',
    'dev_4':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04',
    'dev_5':   'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05',
    'dev_zip': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip',
    'eval_zip':'https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip',
    'gt_zip':  'https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip',
}

SAMPLE_RATE     = 16000
# Soglie dal config ufficiale mae_ast.yaml
MAX_SAMPLE_SIZE = 250000   # 15.6 secondi a 16kHz
MIN_SAMPLE_SIZE = 32000    # 2 secondi a 16kHz


def scarica_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  Gia presente: {os.path.basename(dest)}")
        return
    print(f"  Scarico: {os.path.basename(dest)} ...")
    subprocess.run(['wget', '-q', '--show-progress', '-O', dest, url], check=True)
    print(f"  OK: {os.path.basename(dest)}")


def converti_wav_in_flac(audio_dir: str) -> int:
    """
    Converte tutti i .wav in .flac a 16kHz mono.

    Il dataloader fairseq del repo MAE-AST accetta .flac e .mkv
    ma non .wav direttamente (il RawAudioDataset di fairseq
    usa soundfile che ha supporto limitato per .wav su alcune piattaforme).
    Usiamo .flac che e lossless e universalmente supportato.

    Usa ffmpeg con resample a 16kHz e mono contemporaneamente
    alla conversione — piu efficiente che due passaggi separati.
    """
    file_wav  = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    file_flac = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]

    if not file_wav:
        print(f"  Nessun .wav trovato (forse gia convertiti: {len(file_flac)} .flac)")
        return len(file_flac)

    if len(file_flac) >= len(file_wav):
        print(f"  Conversione gia fatta: {len(file_flac)} .flac presenti")
        return len(file_flac)

    print(f"  Converto {len(file_wav)} .wav -> .flac (16kHz mono)...")
    print(f"  Tempo stimato: ~5-10 minuti su Colab")

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
                '-ar', str(SAMPLE_RATE),
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
    """
    Conta i campioni audio grezzi in un file .flac.

    IMPORTANTE: il TSV deve contenere campioni audio GREZZI a 16kHz,
    non frame fbank. Il task mae_ast_pretraining calcola le fbank
    internamente durante il training.

    Usa torchaudio.info() che legge solo l'header — veloce su 51k file.
    I file sono gia stati convertiti a 16kHz quindi non serve resample.
    """
    import torchaudio
    info = torchaudio.info(percorso)
    return info.num_frames


def genera_tsv(
    audio_dir: str,
    output_path: str,
) -> int:
    """
    Genera file TSV nel formato richiesto da fairseq mae_ast_pretraining.

    Formato (verificato dal README MAE-AST e dal codice del task):
        /
        /percorso/assoluto/file1.flac\tn_campioni_audio_16kHz
        /percorso/assoluto/file2.flac\tn_campioni_audio_16kHz

    Applica i filtri del config ufficiale:
        min_sample_size: 32000  (2 secondi)
        max_sample_size: 250000 (15.6 secondi)
    """
    audio_dir = os.path.abspath(audio_dir)
    file_flac = sorted([f for f in os.listdir(audio_dir) if f.endswith('.flac')])

    if not file_flac:
        print(f"  Nessun .flac in: {audio_dir}")
        return 0

    print(f"  Analizzo {len(file_flac)} file .flac...")

    incluse  = 0
    escluse  = 0
    troncate = 0
    errori   = 0
    righe    = []

    for i, nome in enumerate(file_flac):
        percorso = os.path.join(audio_dir, nome)
        try:
            n = conta_campioni_flac(percorso)

            # Filtra clip troppo corte (sotto min_sample_size del config)
            if n < MIN_SAMPLE_SIZE:
                escluse += 1
                continue

            # Clip troppo lunghe: tronca al max (il task fa random_crop)
            if n > MAX_SAMPLE_SIZE:
                n = MAX_SAMPLE_SIZE
                troncate += 1

            righe.append(f"{percorso}\t{n}")
            incluse += 1

        except Exception as e:
            errori += 1
            if errori <= 5:
                print(f"  Errore su {nome}: {e}")

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(file_flac)} ...")

    # Prima riga: "/" come da README MAE-AST
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('/\n')
        f.write('\n'.join(righe) + '\n')

    print(f"  TSV: {output_path}")
    print(f"    Incluse:  {incluse}")
    print(f"    Escluse:  {escluse}  (sotto {MIN_SAMPLE_SIZE} campioni = 2s)")
    print(f"    Troncate: {troncate} (sopra {MAX_SAMPLE_SIZE} campioni = 15.6s)")
    print(f"    Errori:   {errori}")

    return incluse


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.audio_dir,  exist_ok=True)

    dev_dir  = os.path.join(args.audio_dir, 'FSD50K.dev_audio')
    eval_dir = os.path.join(args.audio_dir, 'FSD50K.eval_audio')

    # Fase 1: Download
    if not args.skip_download:
        print("\n[1/5] Download FSD50K da Zenodo (~25 GB)...")
        dl_dir = os.path.join(args.audio_dir, 'zips')
        os.makedirs(dl_dir, exist_ok=True)
        for nome, url in ZENODO_URLS.items():
            scarica_file(url, os.path.join(dl_dir, os.path.basename(url)))
    else:
        print("\n[1/5] Download saltato")
        dl_dir = os.path.join(args.audio_dir, 'zips')

    # Fase 2: Estrazione
    if not args.skip_extract:
        print("\n[2/5] Estrazione zip...")
        import shutil

        dev_zip = os.path.join(dl_dir, 'FSD50K.dev_audio.zip')
        n_dev = len([f for f in os.listdir(dev_dir)
                     if f.endswith(('.wav', '.flac'))]) if os.path.exists(dev_dir) else 0
        if n_dev < 1000:
            print("  Estraggo FSD50K.dev_audio (zip multi-volume)...")
            subprocess.run(['unzip', '-q', dev_zip, '-d', args.audio_dir], check=True)
            print("  OK dev_audio")
        else:
            print(f"  dev_audio gia estratto ({n_dev} file)")

        eval_zip = os.path.join(dl_dir, 'FSD50K.eval_audio.zip')
        n_eval = len([f for f in os.listdir(eval_dir)
                      if f.endswith(('.wav', '.flac'))]) if os.path.exists(eval_dir) else 0
        if n_eval < 100:
            print("  Estraggo FSD50K.eval_audio...")
            subprocess.run(['unzip', '-q', eval_zip, '-d', args.audio_dir], check=True)
            print("  OK eval_audio")
        else:
            print(f"  eval_audio gia estratto ({n_eval} file)")

        # Elimina zip per liberare ~25 GB
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
            print("  Zip eliminati (liberati ~25 GB)")
    else:
        print("\n[2/5] Estrazione saltata")

    # Fase 3: Conversione wav -> flac
    if not args.skip_convert:
        print("\n[3/5] Conversione .wav -> .flac (16kHz mono)...")
        print("  dev_audio:")
        converti_wav_in_flac(dev_dir)
        print("  eval_audio:")
        converti_wav_in_flac(eval_dir)
    else:
        print("\n[3/5] Conversione saltata")

    # Fase 4: Verifica
    print("\n[4/5] Verifica file...")
    n_dev  = len([f for f in os.listdir(dev_dir)  if f.endswith('.flac')]) if os.path.exists(dev_dir)  else 0
    n_eval = len([f for f in os.listdir(eval_dir) if f.endswith('.flac')]) if os.path.exists(eval_dir) else 0
    print(f"  dev_audio:  {n_dev} .flac  (atteso: ~40.966)")
    print(f"  eval_audio: {n_eval} .flac  (atteso: ~10.231)")
    if n_dev < 1000:
        print("  ERRORE: dev_audio sembra vuoto")
        return

    # Fase 5: Genera TSV
    print("\n[5/5] Generazione TSV...")
    train_tsv = os.path.join(args.output_dir, 'train.tsv')
    valid_tsv = os.path.join(args.output_dir, 'valid.tsv')

    print("\n  train.tsv:")
    n_train = genera_tsv(dev_dir, train_tsv)
    print("\n  valid.tsv:")
    n_valid = genera_tsv(eval_dir, valid_tsv)

    # Verifica formato
    with open(train_tsv) as f:
        prima = f.readline().strip()
        seconda = f.readline().strip()
    print(f"\n  Prima riga:  '{prima}'  (atteso: '/')")
    print(f"  Esempio:     '{seconda[:70]}'")
    assert prima == '/', f"ERRORE formato TSV: prima riga deve essere '/', trovato '{prima}'"

    print(f"\n{'='*55}")
    print(f"  TSV PRONTI PER fairseq-hydra-train")
    print(f"{'='*55}")
    print(f"  train.tsv: {n_train} clip")
    print(f"  valid.tsv: {n_valid} clip")
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
