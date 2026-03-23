"""
src/dataset.py
==============
Classe Dataset PyTorch per ESC-50.

In PyTorch, un Dataset è un oggetto che sa rispondere a due domande:
  1. Quanti campioni ho in totale? → __len__()
  2. Dammi il campione numero N     → __getitem__(n)

Il DataLoader poi usa questo oggetto per costruire i batch
(gruppi di campioni) da dare alla rete durante il training.

Questo file dipende da preprocessing.py per la conversione audio.
Non sa niente del modello — sa solo caricare e preparare i dati.
"""

import os
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

# Importa la funzione di preprocessing dallo stesso package src/
from src.preprocessing import load_audio, N_MELS, TARGET_FRAMES


class ESC50Dataset(Dataset):
    """
    Dataset PyTorch per ESC-50 (Environmental Sound Classification).

    ESC-50 contiene 2000 clip audio da 5 secondi, divise in
    50 classi (cane, pioggia, sirena, applausi, ecc.).
    Il dataset è già diviso in 5 fold predefiniti — questa classe
    accetta una lista di fold da includere, permettendo di creare
    facilmente i set di training e test per la cross-validation.

    Struttura attesa su disco:
        ESC-50/
        ├── audio/           ← file .wav
        │   ├── 1-100032-A-0.wav
        │   ├── ...
        └── meta/
            └── esc50.csv    ← metadati: filename, fold, target, category

    Il CSV ha queste colonne rilevanti:
        filename  → nome del file .wav (es. "1-100032-A-0.wav")
        fold      → numero del fold (1-5)
        target    → etichetta numerica (0-49)
        category  → nome della classe (es. "dog")
    """

    def __init__(
            self,
            csv_path: str,
            audio_dir: str,
            folds: list,
            augment: bool = False,
    ):
        """
        Inizializza il dataset.

        Args:
            csv_path:  Percorso al file esc50.csv
                       Es: "/data/ESC-50/meta/esc50.csv"
            audio_dir: Cartella che contiene i file .wav
                       Es: "/data/ESC-50/audio"
            folds:     Lista di fold da includere.
                       Training tipico: [1, 2, 3, 4]
                       Test tipico:     [5]
            augment:   Se True, applica SpecAugment durante il caricamento.
                       Usare True solo per il training, mai per il test.
        """
        self.audio_dir = audio_dir
        self.augment   = augment

        # ── Carica e filtra il CSV ────────────────────────────────────
        # pandas legge il CSV in un DataFrame (tabella in memoria)
        df = pd.read_csv(csv_path)

        # Tieni solo le righe i cui fold sono nella lista richiesta.
        # Esempio: folds=[1,2,3,4] → tieni 1600 righe su 2000
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        # reset_index(drop=True) rinumera le righe da 0 dopo il filtraggio

        # Estrai le colonne che ci servono come liste Python
        self.file_names = df["filename"].tolist()   # nomi file .wav
        self.labels     = df["target"].tolist()     # etichette 0-49
        self.categories = df["category"].tolist()   # nomi classi (per debug)

        # Dizionario indice → nome classe (utile per stampare predizioni)
        # Es: {0: "dog", 1: "rooster", ...}
        self.idx_to_class = dict(zip(df["target"], df["category"]))

        print(f"  ESC50Dataset: {len(self.file_names)} clip "
              f"| fold(s): {folds} "
              f"| augment: {augment}")

    def __len__(self) -> int:
        """
        Restituisce il numero totale di campioni nel dataset.

        Il DataLoader chiama questo metodo per sapere quanti
        batch creare in un'epoca.
        """
        return len(self.file_names)

    def __getitem__(self, idx: int):
        """
        Carica e restituisce il campione all'indice idx.

        Questo metodo viene chiamato automaticamente dal DataLoader,
        migliaia di volte per epoca. Deve essere veloce e robusto.

        Args:
            idx: Indice del campione (0 → len-1)

        Returns:
            Tupla (spettrogramma, etichetta):
                spettrogramma: Tensor [1, 128, 512]  float32
                etichetta:     Tensor []             int64 (long)
        """

        # ── Costruisci il percorso completo al file audio ─────────────
        file_name = self.file_names[idx]
        percorso  = os.path.join(self.audio_dir, file_name)

        # ── Carica e preprocessa l'audio ─────────────────────────────
        # load_audio() fa tutto: caricamento, mono, resample,
        # spettrogramma mel, log, normalizzazione, padding/troncamento
        # Restituisce Tensor [1, 128, 512]
        spettrogramma = load_audio(percorso)

        # ── Data Augmentation (solo se augment=True, solo training) ──
        # SpecAugment maschera casualmente bande di frequenza e frame
        # temporali per ridurre l'overfitting su ESC-50 (dataset piccolo)
        if self.augment:
            spettrogramma = self._spec_augment(spettrogramma)

        # ── Prepara l'etichetta ────────────────────────────────────────
        # L'etichetta deve essere int64 (LongTensor) perché
        # CrossEntropyLoss si aspetta esattamente questo tipo
        etichetta = torch.tensor(self.labels[idx], dtype=torch.long)

        return spettrogramma, etichetta

    def _spec_augment(
            self,
            spec: torch.Tensor,
            freq_mask_max: int = 24,
            time_mask_max: int = 48,
    ) -> torch.Tensor:
        """
        Applica SpecAugment allo spettrogramma.

        SpecAugment (Park et al. 2019) è la tecnica di augmentation
        standard per l'audio. Funziona così:
          - Maschera frequenze: sceglie un intervallo casuale
            sull'asse Y e lo azzera (simula frequenze mancanti)
          - Maschera tempo: sceglie un intervallo casuale
            sull'asse X e lo azzera (simula audio corrotto)

        L'effetto è che il modello impara a classificare anche
        quando parte dell'informazione manca — diventa più robusto.

        Args:
            spec:          Tensor [1, N_MELS, TARGET_FRAMES]
            freq_mask_max: Numero massimo di bande mel da mascherare
                           (paper usa 24 su 128 = ~19%)
            time_mask_max: Numero massimo di frame da mascherare
                           (paper usa ~10% di TARGET_FRAMES)

        Returns:
            Tensor [1, N_MELS, TARGET_FRAMES] con maschere applicate
        """
        spec = spec.clone()   # non modificare il tensore originale
        _, n_mels, n_frames = spec.shape

        # ── Maschera frequenze ────────────────────────────────────────
        # Scegli quante bande mascherare (da 0 a freq_mask_max)
        f = random.randint(0, freq_mask_max)
        if f > 0:
            # Scegli da quale banda iniziare
            f0 = random.randint(0, n_mels - f)
            # Azzera le bande da f0 a f0+f
            spec[:, f0 : f0 + f, :] = 0.0

        # ── Maschera tempo ────────────────────────────────────────────
        t = random.randint(0, time_mask_max)
        if t > 0:
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0 : t0 + t] = 0.0

        return spec

    def get_class_name(self, idx: int) -> str:
        """
        Restituisce il nome della classe per un indice numerico.
        Utile per stampare predizioni in modo leggibile.

        Es: get_class_name(0) → "dog"
        """
        return self.idx_to_class.get(idx, f"classe_{idx}")


# ──────────────────────────────────────────────────────────────────────
# TEST — eseguibile con: python src/dataset.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    print("=" * 55)
    print("  TEST: src/dataset.py")
    print("=" * 55)

    # ── Cerca ESC-50 ─────────────────────────────────────────────────
    # Controlla percorsi comuni dove potrebbe essere il dataset
    candidati = [
        "ESC-50",
        "../ESC-50",
        os.path.expanduser("~/ESC-50"),
        "data/ESC-50",
    ]

    esc50_dir = None
    for c in candidati:
        if os.path.exists(os.path.join(c, "meta", "esc50.csv")):
            esc50_dir = c
            break

    if esc50_dir is None:
        print("\n  ⚠️  ESC-50 non trovato in locale.")
        print("  Questo è normale — ESC-50 verrà scaricato su Colab.")
        print()
        print("  Per testare in locale, scarica ESC-50:")
        print("  git clone https://github.com/karolpiczak/ESC-50")
        print()
        print("  Poi esegui di nuovo questo script.")
        print("  Alternativa: passiamo al passo successivo (src/model.py)")
        print("=" * 55)
        sys.exit(0)

    csv_path  = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    # ── Crea i dataset per fold 1-4 (train) e fold 5 (test) ──────────
    print(f"\n  ESC-50 trovato: {esc50_dir}")
    print("\n  Creazione dataset...")

    ds_train = ESC50Dataset(
        csv_path  = csv_path,
        audio_dir = audio_dir,
        folds     = [1, 2, 3, 4],
        augment   = True,
    )

    ds_test = ESC50Dataset(
        csv_path  = csv_path,
        audio_dir = audio_dir,
        folds     = [5],
        augment   = False,
    )

    # ── Verifica dimensioni ───────────────────────────────────────────
    print(f"\n  Campioni training (fold 1-4): {len(ds_train)}")
    print(f"  Campioni test     (fold 5):   {len(ds_test)}")

    assert len(ds_train) == 1600, f"Attesi 1600, trovati {len(ds_train)}"
    assert len(ds_test)  ==  400, f"Attesi 400,  trovati {len(ds_test)}"

    # ── Carica un singolo campione ────────────────────────────────────
    print("\n  Caricamento primo campione...")
    spec, label = ds_train[0]

    print(f"  Shape spettrogramma: {spec.shape}   (atteso: [1, 128, 512])")
    print(f"  Dtype spettrogramma: {spec.dtype}   (atteso: float32)")
    print(f"  Etichetta:           {label.item()}  "
          f"({ds_train.get_class_name(label.item())})")
    print(f"  Min / Max:           {spec.min():.3f} / {spec.max():.3f}")

    assert spec.shape == (1, 128, 512)
    assert spec.dtype == torch.float32
    assert label.dtype == torch.long

    # ── Crea un DataLoader e verifica un batch ────────────────────────
    print("\n  Test DataLoader (batch_size=4)...")
    loader = DataLoader(ds_train, batch_size=4, shuffle=True)

    batch_spec, batch_labels = next(iter(loader))
    print(f"  Shape batch spettrogrammi: {batch_spec.shape}"
          f"  (atteso: [4, 1, 128, 512])")
    print(f"  Shape batch etichette:     {batch_labels.shape}"
          f"  (atteso: [4])")
    print(f"  Classi nel batch:          "
          f"{[ds_train.get_class_name(l.item()) for l in batch_labels]}")

    assert batch_spec.shape   == (4, 1, 128, 512)
    assert batch_labels.shape == (4,)

    print(f"\n  ✅ Tutti i controlli superati!")
    print("=" * 55)