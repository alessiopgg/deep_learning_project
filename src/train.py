"""
src/train.py
============
Funzioni di training e valutazione per il fine-tuning su ESC-50.

Questo modulo contiene due funzioni principali:

  train_one_epoch() — esegue un'epoca di training completa
  evaluate()        — valuta il modello su un set di dati

Entrambe sono generiche: non sanno niente di ESC-50 nello specifico.
Funzionano con qualsiasi modello e dataloader PyTorch.
Vengono chiamate dal notebook PartB_finetune_ESC50.ipynb.

GLOSSARIO RAPIDO:
  epoca     = un passaggio completo su tutti i campioni di training
  batch     = gruppo di campioni processati insieme (es. 24 clip)
  loss      = quanto è sbagliata la predizione (numero da minimizzare)
  gradiente = direzione in cui modificare i pesi per ridurre la loss
  optimizer = algoritmo che aggiorna i pesi usando i gradienti
  scheduler = algoritmo che modifica il learning rate nel tempo
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
        modello: nn.Module,
        loader: DataLoader,
        ottimizzatore: torch.optim.Optimizer,
        criterio: nn.Module,
        device: torch.device,
        epoch: int,
        n_epochs: int,
        log_ogni: int = 20,
) -> dict:
    """
    Esegue una singola epoca di training.

    Una epoca = il modello vede ogni campione del training set
    esattamente una volta, suddiviso in batch.

    Per ogni batch:
      1. Sposta i dati sulla GPU
      2. Forward pass: calcola le predizioni
      3. Calcola la loss: quanto sono sbagliate le predizioni
      4. Backward pass: calcola i gradienti (dL/dW per ogni peso W)
      5. Gradient clipping: limita i gradienti per stabilità
      6. Aggiorna i pesi: W = W - lr × gradiente

    Args:
        modello:      Il modello PyTorch (MAEASTFineTune)
        loader:       DataLoader del training set
        ottimizzatore: Es. Adam con due gruppi di LR
        criterio:     Funzione di loss (CrossEntropyLoss)
        device:       torch.device('cuda') o torch.device('cpu')
        epoch:        Numero dell'epoca corrente (per il logging)
        n_epochs:     Numero totale di epoche (per il logging)
        log_ogni:     Stampa il progresso ogni N batch

    Returns:
        Dizionario con metriche dell'epoca:
          'loss_media'  → loss media su tutti i batch
          'accuracy'    → percentuale predizioni corrette sul training
          'tempo_sec'   → secondi impiegati per l'epoca
    """

    # ── Metti il modello in modalità training ─────────────────────────
    # modello.train() attiva due meccanismi importanti:
    #   1. Dropout: disattiva casualmente alcuni neuroni (regolarizzazione)
    #   2. BatchNorm: usa le statistiche del batch corrente
    # Senza questa chiamata, il modello si comporta come in valutazione
    # e il dropout non funziona — risultato: overfitting
    modello.train()

    # Accumulatori per le metriche
    loss_totale  = 0.0
    corretti     = 0
    totale       = 0
    t_inizio     = time.time()
    n_batch      = len(loader)

    for i, (spettrogrammi, etichette) in enumerate(loader):

        # ── Step 1: sposta i dati sulla GPU ───────────────────────────
        # I dati vengono caricati in RAM dal DataLoader.
        # .to(device) li copia sulla VRAM della GPU.
        # Non spostare i dati sulla GPU è uno degli errori più comuni —
        # causerebbe un errore "Expected device cuda but got cpu".
        spettrogrammi = spettrogrammi.to(device)
        etichette     = etichette.to(device)

        # ── Step 2: forward pass ──────────────────────────────────────
        # Calcola le predizioni del modello.
        # logits: [batch, 50] — uno score per ogni classe
        # Non sono probabilità — CrossEntropyLoss applica softmax
        # internamente, quindi non devi applicarlo tu
        logits = modello(spettrogrammi)

        # ── Step 3: calcola la loss ───────────────────────────────────
        # CrossEntropyLoss misura quanto le predizioni sono lontane
        # dalle etichette reali.
        # Internamente fa: softmax(logits) → log → negative log likelihood
        # Più la loss è alta, più le predizioni sono sbagliate.
        loss = criterio(logits, etichette)

        # ── Step 4: backward pass ─────────────────────────────────────
        # Prima di calcolare i nuovi gradienti, azzera quelli precedenti.
        # PyTorch accumula i gradienti per default — senza zero_grad()
        # i gradienti si sommerebbero tra un batch e l'altro (bug comune).
        ottimizzatore.zero_grad()

        # loss.backward() calcola dL/dW per ogni parametro W del modello.
        # Usa la chain rule (regola della catena) all'indietro attraverso
        # tutti i layer — da qui il nome "backpropagation".
        loss.backward()

        # ── Step 5: gradient clipping ─────────────────────────────────
        # Limita la norma dei gradienti a max_norm=1.0.
        # Senza clipping, gradienti molto grandi possono fare "esplodere"
        # i pesi in un singolo step — il training diventa instabile.
        # Per i transformer è una pratica standard.
        torch.nn.utils.clip_grad_norm_(
            modello.parameters(),
            max_norm=1.0
        )

        # ── Step 6: aggiorna i pesi ───────────────────────────────────
        # L'ottimizzatore usa i gradienti calcolati da backward()
        # per aggiornare i pesi: W = W - lr × gradiente
        # Adam usa anche la storia dei gradienti precedenti per
        # aggiornamenti più stabili.
        ottimizzatore.step()

        # ── Accumula metriche ─────────────────────────────────────────
        loss_totale += loss.item()
        # .item() converte un tensore scalare in un float Python

        # Calcola quante predizioni sono corrette in questo batch
        _, predizioni = torch.max(logits, dim=1)
        # torch.max restituisce (valori_massimi, indici_massimi)
        # gli indici sono le classi predette (0-49)
        corretti += (predizioni == etichette).sum().item()
        totale   += etichette.size(0)

        # ── Log del progresso ─────────────────────────────────────────
        if (i + 1) % log_ogni == 0 or (i + 1) == n_batch:
            loss_corrente = loss_totale / (i + 1)
            acc_corrente  = corretti / totale * 100
            elapsed       = time.time() - t_inizio
            print(
                f"  Epoch {epoch:2d}/{n_epochs} | "
                f"Batch {i+1:3d}/{n_batch} | "
                f"Loss: {loss_corrente:.4f} | "
                f"Acc: {acc_corrente:.1f}% | "
                f"Tempo: {elapsed:.0f}s",
                end='\r'
            )

    print()  # va a capo dopo l'ultimo \r

    tempo_totale = time.time() - t_inizio

    return {
        'loss_media': loss_totale / n_batch,
        'accuracy':   corretti / totale,
        'tempo_sec':  tempo_totale,
    }


def evaluate(
        modello: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterio: nn.Module = None,
) -> dict:
    """
    Valuta il modello su un set di dati senza aggiornare i pesi.

    Usata sia per il validation set durante il training
    che per il test set finale.

    La differenza principale rispetto a train_one_epoch:
      - modello.eval() disabilita dropout e BatchNorm in train mode
      - torch.no_grad() disabilita il calcolo dei gradienti
        (non servono in valutazione, risparmia memoria e tempo ~30%)

    Args:
        modello:   Il modello PyTorch
        loader:    DataLoader del validation/test set
        device:    torch.device('cuda') o torch.device('cpu')
        criterio:  Funzione di loss (opzionale — se None non calcola loss)

    Returns:
        Dizionario con metriche:
          'accuracy'   → percentuale predizioni corrette (0.0 - 1.0)
          'loss_media' → loss media (None se criterio non fornito)
          'n_campioni' → numero totale di campioni valutati
    """

    # ── Metti il modello in modalità valutazione ──────────────────────
    # modello.eval() disabilita:
    #   - Dropout: tutti i neuroni attivi (predizioni deterministiche)
    #   - BatchNorm: usa le statistiche globali invece del batch
    # Senza questa chiamata i risultati sarebbero diversi ad ogni run
    # (per via del dropout casuale) — non confrontabili
    modello.eval()

    corretti    = 0
    totale      = 0
    loss_totale = 0.0
    n_batch     = len(loader)

    # torch.no_grad() disabilita il grafo computazionale per i gradienti
    # Non ci serve calcolare gradienti in valutazione — risparmia:
    #   - ~30% di tempo di esecuzione
    #   - memoria VRAM (non tiene i tensori intermedi per backward)
    with torch.no_grad():

        for spettrogrammi, etichette in loader:

            spettrogrammi = spettrogrammi.to(device)
            etichette     = etichette.to(device)

            # Forward pass: solo predizioni, nessun aggiornamento pesi
            logits = modello(spettrogrammi)

            # Calcola la loss se il criterio è fornito
            if criterio is not None:
                loss         = criterio(logits, etichette)
                loss_totale += loss.item()

            # La classe predetta è quella con lo score più alto
            _, predizioni = torch.max(logits, dim=1)

            corretti += (predizioni == etichette).sum().item()
            totale   += etichette.size(0)

    accuracy   = corretti / totale
    loss_media = (loss_totale / n_batch) if criterio is not None else None

    return {
        'accuracy':   accuracy,
        'loss_media': loss_media,
        'n_campioni': totale,
    }


def cross_validation_5fold(
        checkpoint_path: str,
        csv_path: str,
        audio_dir: str,
        config: dict,
        device: torch.device,
) -> dict:
    """
    Esegue la 5-fold cross-validation completa su ESC-50.

    Questo è il protocollo ufficiale di ESC-50 — l'unico modo
    per ottenere risultati confrontabili con il paper.

    Per ogni fold (1→5):
      - Crea training set (4 fold) e test set (1 fold)
      - Inizializza il modello con i pesi MAE-AST pretrainati
      - Esegue N epoche di fine-tuning
      - Valuta sul test set e salva la best accuracy

    Risultato finale: media ± std delle 5 best accuracy

    Args:
        checkpoint_path: Percorso al .pth convertito
        csv_path:        Percorso a esc50.csv
        audio_dir:       Cartella con i .wav di ESC-50
        config:          Dizionario con tutti gli iperparametri
                         (vedi config/finetune.yaml)
        device:          torch.device

    Returns:
        Dizionario con:
          'accuracies'  → lista di 5 float (una per fold)
          'media'       → media delle 5 accuracies
          'std'         → deviazione standard
          'storia'      → dizionario con loss/acc per ogni epoca/fold
    """
    import numpy as np
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.dataset import ESC50Dataset
    from src.model import MAEASTFineTune

    tutti_i_fold    = [1, 2, 3, 4, 5]
    accuracies      = []
    storia_completa = {}
    t_inizio_cv     = time.time()

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS VALIDATION — ESC-50")
    print(f"  Target paper: 90.0% (MAE-AST Patch 12L)")
    print(f"{'='*60}")

    for fold_test in tutti_i_fold:

        fold_train = [f for f in tutti_i_fold if f != fold_test]

        print(f"\n--- FOLD {fold_test}/5 ---")
        print(f"  Training: fold {fold_train}")
        print(f"  Test:     fold [{fold_test}]")

        # ── Dataset e DataLoader ──────────────────────────────────────
        ds_train = ESC50Dataset(
            csv_path  = csv_path,
            audio_dir = audio_dir,
            folds     = fold_train,
            augment   = True,
        )
        ds_test = ESC50Dataset(
            csv_path  = csv_path,
            audio_dir = audio_dir,
            folds     = [fold_test],
            augment   = False,
        )

        loader_train = DataLoader(
            ds_train,
            batch_size  = config['batch_size'],
            shuffle     = True,
            num_workers = config.get('num_workers', 2),
            pin_memory  = True,
            drop_last   = True,
        )
        loader_test = DataLoader(
            ds_test,
            batch_size  = config['batch_size'],
            shuffle     = False,
            num_workers = config.get('num_workers', 2),
            pin_memory  = True,
        )

        # ── Modello — nuovo per ogni fold ────────────────────────────
        # Ogni fold riparte dai pesi MAE-AST pretrainati
        # (non dai pesi del fold precedente)
        modello = MAEASTFineTune(
            checkpoint_path = checkpoint_path,
            num_classes     = config['num_classes'],
            drop_rate       = config.get('drop_rate', 0.3),
        ).to(device)

        # ── Ottimizzatore con LR diversi ─────────────────────────────
        # Encoder: LR basso — pesi già buoni dal pretraining
        # Classificatore: LR alto — deve imparare da zero
        gruppi = modello.get_optimizer_groups(
            lr_encoder        = config['lr_encoder'],
            lr_classificatore = config['lr_classificatore'],
            weight_decay      = config.get('weight_decay', 5e-4),
        )
        ottimizzatore = torch.optim.Adam(gruppi)

        # ── Scheduler cosine annealing ────────────────────────────────
        # Il LR decade seguendo una curva coseno da LR a eta_min.
        # Nelle ultime epoche il LR è molto basso — convergenza fine.
        scheduler = CosineAnnealingLR(
            ottimizzatore,
            T_max   = config['n_epochs'],
            eta_min = config.get('lr_min', 1e-6),
        )

        # ── Loss function ─────────────────────────────────────────────
        # CrossEntropyLoss con label_smoothing:
        # invece di target = [0,0,1,0,...] usa [0.002, 0.002, 0.91, ...]
        # Riduce la confidenza eccessiva e migliora la generalizzazione
        criterio = nn.CrossEntropyLoss(
            label_smoothing = config.get('label_smoothing', 0.1)
        )

        # ── Training loop ─────────────────────────────────────────────
        best_acc  = 0.0
        storia_fold = {'loss': [], 'acc_train': [], 'acc_test': []}
        t_fold_start = time.time()

        for epoch in range(1, config['n_epochs'] + 1):

            # Epoca di training
            metriche_train = train_one_epoch(
                modello       = modello,
                loader        = loader_train,
                ottimizzatore = ottimizzatore,
                criterio      = criterio,
                device        = device,
                epoch         = epoch,
                n_epochs      = config['n_epochs'],
            )

            # Aggiorna il learning rate
            scheduler.step()

            # Valuta ogni val_ogni epoche e all'ultima
            val_ogni = config.get('val_ogni', 5)
            if epoch % val_ogni == 0 or epoch == config['n_epochs']:

                metriche_test = evaluate(
                    modello   = modello,
                    loader    = loader_test,
                    device    = device,
                    criterio  = criterio,
                )

                acc_test = metriche_test['accuracy']
                if acc_test > best_acc:
                    best_acc = acc_test

                lr_attuale = ottimizzatore.param_groups[0]['lr']

                print(
                    f"  Epoch {epoch:2d}/{config['n_epochs']} | "
                    f"Loss: {metriche_train['loss_media']:.4f} | "
                    f"Train: {metriche_train['accuracy']*100:.1f}% | "
                    f"Test: {acc_test*100:.1f}% | "
                    f"Best: {best_acc*100:.1f}% | "
                    f"LR: {lr_attuale:.1e}"
                )

                storia_fold['loss'].append(metriche_train['loss_media'])
                storia_fold['acc_train'].append(metriche_train['accuracy'])
                storia_fold['acc_test'].append(acc_test)

        t_fold = time.time() - t_fold_start
        accuracies.append(best_acc)
        storia_completa[f'fold_{fold_test}'] = storia_fold

        print(f"\n  ✅ Fold {fold_test} completato in {t_fold/60:.1f} min")
        print(f"     Best accuracy: {best_acc*100:.2f}%")

        # Libera memoria GPU prima del prossimo fold
        del modello, ottimizzatore, scheduler
        torch.cuda.empty_cache()

    # ── Risultati finali ──────────────────────────────────────────────
    media = float(np.mean(accuracies))
    std   = float(np.std(accuracies))
    t_tot = time.time() - t_inizio_cv

    print(f"\n{'='*60}")
    print(f"  RISULTATI FINALI — 5-FOLD CROSS VALIDATION")
    print(f"{'='*60}")
    for i, acc in enumerate(accuracies):
        print(f"  Fold {i+1}: {acc*100:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Media:  {media*100:.2f}% ± {std*100:.2f}%")
    print(f"  Target: 90.00% (paper MAE-AST Patch 12L)")
    diff = (media - 0.90) * 100
    segno = '+' if diff >= 0 else ''
    print(f"  Delta:  {segno}{diff:.2f}%")
    print(f"  Tempo totale: {t_tot/60:.1f} min")
    print(f"{'='*60}")

    return {
        'accuracies': accuracies,
        'media':      media,
        'std':        std,
        'storia':     storia_completa,
    }


# ──────────────────────────────────────────────────────────────────────
# TEST — eseguibile con: python src/train.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn

    print("=" * 55)
    print("  TEST: src/train.py")
    print("=" * 55)

    # Crea modello e dati finti per testare le funzioni
    device = torch.device('cpu')

    # Modello minimo per il test
    class ModelloTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(128 * 512, 50)
        def forward(self, x):
            return self.layer(x.view(x.size(0), -1))

    modello   = ModelloTest().to(device)
    criterio  = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=1e-3)

    # DataLoader finto: 3 batch da 4 campioni
    dati_finti = [
        (torch.randn(4, 1, 128, 512), torch.randint(0, 50, (4,)))
        for _ in range(3)
    ]

    print("\n  Test train_one_epoch...")
    metriche = train_one_epoch(
        modello       = modello,
        loader        = dati_finti,
        ottimizzatore = ottimizzatore,
        criterio      = criterio,
        device        = device,
        epoch         = 1,
        n_epochs      = 5,
        log_ogni      = 1,
    )
    print(f"  Loss media:  {metriche['loss_media']:.4f}")
    print(f"  Accuracy:    {metriche['accuracy']*100:.1f}%")
    print(f"  Tempo:       {metriche['tempo_sec']:.2f}s")
    assert 'loss_media' in metriche
    assert 'accuracy'   in metriche

    print("\n  Test evaluate...")
    risultati = evaluate(
        modello  = modello,
        loader   = dati_finti,
        device   = device,
        criterio = criterio,
    )
    print(f"  Accuracy:    {risultati['accuracy']*100:.1f}%")
    print(f"  Loss media:  {risultati['loss_media']:.4f}")
    print(f"  Campioni:    {risultati['n_campioni']}")
    assert 'accuracy'   in risultati
    assert 'n_campioni' in risultati

    print(f"\n  ✅ Tutti i controlli superati!")
    print("=" * 55)