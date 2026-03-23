"""
src/preprocessing.py
====================
Converte un file audio in uno spettrogramma mel normalizzato.

Questo modulo è il mattone fondamentale del progetto: viene chiamato
da dataset.py per ogni clip audio, sia durante il pretraining che
durante il fine-tuning. L'output è sempre un tensore di forma fissa
[1, 128, 512] — pensalo come un'immagine in scala di grigi dove:
  - 1      = numero di canali (come grayscale)
  - 128    = frequenze (asse Y dello spettrogramma)
  - 512    = frame temporali (asse X dello spettrogramma)

Parametri scelti per replicare esattamente la configurazione
del paper MAE-AST / SSAST (Baade et al. 2022, Gong et al. 2022).
"""

import torch
import torchaudio
import torchaudio.transforms as T

# ──────────────────────────────────────────────────────────────────────
# COSTANTI — parametri dello spettrogramma
# Questi valori non cambiano mai nel progetto.
# Altri moduli possono importarli direttamente da qui se necessario.
# ──────────────────────────────────────────────────────────────────────

# Frequenza di campionamento target.
# Tutti i file audio vengono ricampionati a questa frequenza,
# indipendentemente dalla loro frequenza originale.
SAMPLE_RATE = 16000

# Numero di filtri mel = altezza dello spettrogramma output.
# Determina la risoluzione sull'asse delle frequenze.
N_MELS = 128

# Dimensione della finestra FFT in campioni.
# Controlla il trade-off tra risoluzione temporale e frequenziale.
N_FFT = 1024

# Numero di campioni tra una finestra FFT e la successiva.
# Con SAMPLE_RATE=16000 e HOP_LENGTH=160:
#   un frame = 160/16000 = 10ms  →  100 frame per secondo
HOP_LENGTH = 160

# Numero di frame temporali target = larghezza dello spettrogramma output.
# Con HOP_LENGTH=160 e SAMPLE_RATE=16000:
#   512 frame × 10ms = 5.12 secondi di audio coperti
# Le clip ESC-50 durano 5 secondi → copertura quasi perfetta senza padding
TARGET_FRAMES = 512

# Statistiche di normalizzazione calcolate su AudioSet da Gong et al. (SSAST).
# Vengono usate anche con FSD50K perché sono lo standard della letteratura
# per modelli audio pretrainati su AudioSet — evita un passaggio di
# calcolo su tutto il dataset di pretraining.
NORM_MEAN = -4.2677393
NORM_STD  =  4.5689974


# ──────────────────────────────────────────────────────────────────────
# FUNZIONE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> torch.Tensor:
    """
    Carica un file audio e lo converte in spettrogramma mel normalizzato.

    Pipeline:
        1. Carica il file audio (WAV, FLAC, MP3, ...)
        2. Converte in mono se stereo
        3. Ricampiona a SAMPLE_RATE se necessario
        4. Calcola lo spettrogramma mel
        5. Applica logaritmo (scala dB)
        6. Normalizza con statistiche AudioSet
        7. Padding o troncamento a TARGET_FRAMES

    Args:
        path: Percorso al file audio. Accetta qualsiasi formato
              supportato da torchaudio (WAV, FLAC, OGG, MP3).

    Returns:
        Tensore di forma [1, N_MELS, TARGET_FRAMES] = [1, 128, 512].
        Dtype: float32.

    Raises:
        FileNotFoundError: Se il file non esiste.
        RuntimeError: Se il file è corrotto o in formato non supportato.
    """

    # ── Step 1: Carica il file audio ──────────────────────────────────
    # torchaudio.load restituisce:
    #   waveform: tensore [n_canali, n_campioni]
    #   sr:       frequenza di campionamento originale del file
    waveform, sr = torchaudio.load(path)
    # waveform shape: [C, T]  dove C = canali, T = campioni

    # ── Step 2: Converti in mono ──────────────────────────────────────
    # Se il file ha più canali (es. stereo = 2 canali), li mediamo.
    # La media mantiene il volume complessivo e non taglia informazione.
    # Se è già mono (C=1), questa operazione non cambia nulla.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # waveform shape: [1, T]

    # ── Step 3: Ricampiona a SAMPLE_RATE ─────────────────────────────
    # ESC-50 è a 44.100 Hz, FSD50K è a 44.100 Hz.
    # Il modello si aspetta 16.000 Hz.
    # Resample gestisce questa conversione in modo efficiente.
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)
    # waveform shape: [1, T_new]  dove T_new = T * (SAMPLE_RATE / sr)

    # ── Step 4: Spettrogramma mel ─────────────────────────────────────
    # MelSpectrogram esegue in sequenza:
    #   a) STFT (Short-Time Fourier Transform) con finestra N_FFT
    #   b) Mappatura sulle bande mel (rispecchia la percezione umana)
    #
    # norm='slaney' e mel_scale='slaney' sono le impostazioni usate
    # nel paper SSAST — importante mantenerle per compatibilità.
    mel_transform = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft       = N_FFT,
        hop_length  = HOP_LENGTH, 
        n_mels      = N_MELS,
        norm        = 'slaney',
        mel_scale   = 'slaney',
    )
    spectrogram = mel_transform(waveform)
    # spectrogram shape: [1, N_MELS, T_frames]
    # T_frames ≈ n_campioni / HOP_LENGTH

    # ── Step 5: Logaritmo ─────────────────────────────────────────────
    # Lo spettrogramma mel grezzo ha valori di energia (positivi, grandi).
    # La scala logaritmica comprime il range dinamico e avvicina la
    # rappresentazione alla percezione umana del volume (in dB).
    # Aggiungiamo 1e-6 per evitare log(0) = -infinito nelle zone silenziose.
    spectrogram = torch.log(spectrogram + 1e-6)
    # spectrogram shape: [1, N_MELS, T_frames]  — stessa forma, valori in log

    # ── Step 6: Normalizzazione ────────────────────────────────────────
    # Porta i valori vicino a media 0, std 1 usando le statistiche
    # calcolate su AudioSet dagli autori di SSAST.
    # Il divisore è NORM_STD * 2 (non NORM_STD) per seguire
    # esattamente la formula del paper originale.
    spectrogram = (spectrogram - NORM_MEAN) / (NORM_STD * 2)
    # spectrogram shape: [1, N_MELS, T_frames]

    # ── Step 7: Padding o troncamento a TARGET_FRAMES ─────────────────
    # La rete si aspetta sempre esattamente TARGET_FRAMES = 512 frame.
    # Se la clip è più corta → padding con zeri a destra
    # Se la clip è più lunga → troncamento (prendi i primi 512 frame)
    n_frames = spectrogram.shape[2]

    if n_frames < TARGET_FRAMES:
        # Padding: aggiunge (TARGET_FRAMES - n_frames) colonne di zeri
        # F.pad vuole il padding nell'ordine (ultimo_dim_destra, ultimo_dim_sinistra, ...)
        pad_amount  = TARGET_FRAMES - n_frames
        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))

    elif n_frames > TARGET_FRAMES:
        # Troncamento: teniamo solo i primi TARGET_FRAMES frame
        spectrogram = spectrogram[:, :, :TARGET_FRAMES]

    # spectrogram shape finale: [1, 128, 512]
    return spectrogram


# ──────────────────────────────────────────────────────────────────────
# TEST — eseguibile direttamente con: python src/preprocessing.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    print("=" * 55)
    print("  TEST: src/preprocessing.py")
    print("=" * 55)

    # Cerca un file WAV nella cartella corrente o nelle sottocartelle
    # Se non ne trovi, crea un segnale sintetico per testare la pipeline
    test_file = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                test_file = os.path.join(root, f)
                break
        if test_file:
            break

    if test_file:
        print(f"\n  File trovato: {test_file}")
        spec = load_audio(test_file)
    else:
        print("\n  Nessun file audio trovato.")
        print("  Creo un segnale sintetico di test (1 secondo, 440 Hz)...")

        # Crea un tono sinusoidale a 440 Hz (La4) come segnale di test
        duration  = 5.0   # secondi
        frequency = 440.0 # Hz
        t         = torch.linspace(0, duration, int(SAMPLE_RATE * duration))
        sine_wave = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        # sine_wave shape: [1, 80000]

        # Salva temporaneamente come WAV per testare il caricamento reale
        test_file = "test_sine_440hz.wav"
        torchaudio.save(test_file, sine_wave, SAMPLE_RATE)
        print(f"  Salvato: {test_file}")

        spec = load_audio(test_file)

        # Rimuovi il file temporaneo
        os.remove(test_file)

    # ── Verifica i risultati ──────────────────────────────────────────
    print(f"\n  RISULTATI:")
    print(f"  Shape output:    {spec.shape}  (atteso: [1, 128, 512])")
    print(f"  Dtype:           {spec.dtype}  (atteso: torch.float32)")
    print(f"  Valore minimo:   {spec.min():.4f}")
    print(f"  Valore massimo:  {spec.max():.4f}")
    print(f"  Media:           {spec.mean():.4f}  (atteso: vicino a 0)")
    print(f"  Std:             {spec.std():.4f}   (atteso: vicino a 0.5)")

    # Controlla che la shape sia corretta
    assert spec.shape == (1, 128, 512), \
        f"ERRORE: shape attesa [1, 128, 512], ottenuta {spec.shape}"
    assert spec.dtype == torch.float32, \
        f"ERRORE: dtype atteso float32, ottenuto {spec.dtype}"

    print(f"\n  ✅ Tutti i controlli superati!")
    print("=" * 55)