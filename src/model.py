"""
src/model.py
============
Definisce il modello MAE-AST per il fine-tuning su ESC-50.

STRUTTURA DEL MODELLO:
    Il modello ha due componenti:

    1. ENCODER (backbone ViT-base pretrainato)
       - Caricato da timm con la struttura vit_base_patch16_224
       - I pesi vengono sostituiti con quelli del checkpoint MAE-AST
       - Durante il fine-tuning viene aggiornato con LR molto basso
         perché ha già imparato rappresentazioni utili dal pretraining

    2. CLASSIFICATORE (testa MLP aggiunta da noi)
       - LayerNorm → Linear(768→512) → GELU → Dropout → Linear(512→50)
       - Inizializzato con pesi casuali — deve imparare da zero
       - Viene aggiornato con LR più alto dell'encoder

FORWARD PASS:
    spettrogramma [batch, 1, 128, 512]
         ↓
    encoder (ViT-base, 12 layer transformer)
         ↓
    output [batch, n_token+1, 768]  ← sequenza di token con dimensioni
         ↓
    cls_token [batch, 768]          ← solo il token di classificazione
         ↓
    classificatore MLP
         ↓
    logits [batch, 50]              ← uno score per ogni classe ESC-50

NOTA SUL CLS TOKEN:
    Il ViT produce un token speciale chiamato CLS token (indice 0)
    che riassume l'intera sequenza. È quello che usiamo per la
    classificazione — gli altri token rappresentano singole patch.
"""

import torch
import torch.nn as nn
import timm


class MAEASTFineTune(nn.Module):
    """
    Modello MAE-AST adattato per il fine-tuning su classificazione audio.

    Eredita da nn.Module — la classe base di tutti i modelli PyTorch.
    Ogni modello PyTorch deve implementare almeno __init__ e forward.
    """

    def __init__(
            self,
            checkpoint_path: str,
            num_classes: int = 50,
            img_size: tuple = (128, 512),
            drop_rate: float = 0.3,
    ):
        """
        Costruisce il modello e carica i pesi pretrainati.

        Args:
            checkpoint_path: Percorso al .pth convertito
                             (output di scripts/convert_checkpoint.py)
            num_classes:     Numero di classi output.
                             50 per ESC-50, 527 per AudioSet, ecc.
            img_size:        Dimensione input dello spettrogramma.
                             (128, 512) = 128 frequenze × 512 frame
            drop_rate:       Dropout nella testa di classificazione.
                             0.3 = disattiva il 30% dei neuroni
                             durante il training (riduce overfitting)
        """
        super().__init__()
        # super().__init__() chiama il costruttore di nn.Module —
        # obbligatorio in ogni modello PyTorch

        self.num_classes = num_classes

        # ── Costruisce il backbone ViT ─────────────────────────────────
        # timm.create_model crea un ViT-base con la struttura standard.
        # Usiamo 'vit_base_patch16_224' come template architetturale,
        # poi sovrascriviamo img_size e in_chans per adattarlo
        # agli spettrogrammi mel (invece delle immagini RGB).
        #
        # Parametri chiave:
        #   pretrained=False  → non scaricare pesi ImageNet,
        #                        li carichiamo noi dal checkpoint
        #   img_size          → sovrascrive 224×224 con 128×512
        #   in_chans=1        → spettrogramma mono (non RGB a 3 canali)
        #   num_classes=0     → rimuove la testa di classificazione
        #                        originale di timm (ne mettiamo una nostra)
        #   global_pool=''    → disabilita il pooling automatico
        #                        vogliamo accedere ai token raw
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained  = False,
            img_size    = img_size,
            in_chans    = 1,
            num_classes = 0,
            global_pool = '',
        )

        # ── Carica i pesi pretrainati MAE-AST ─────────────────────────
        # Questo è il passaggio critico — sostituisce i pesi casuali
        # dell'encoder appena creato con quelli imparati durante
        # il pretraining MAE-AST su AudioSet + LibriSpeech
        n_loaded = self._carica_pesi(checkpoint_path)

        # ── Testa di classificazione ───────────────────────────────────
        # Prende il CLS token (768 dimensioni) e produce 50 score.
        # Struttura:
        #   LayerNorm  → normalizza le attivazioni (stabilità)
        #   Linear     → 768 → 512 (riduzione dimensionale)
        #   GELU       → funzione di attivazione non lineare
        #                (più smooth di ReLU, standard per transformer)
        #   Dropout    → regolarizzazione (disattiva neuroni random)
        #   Linear     → 512 → num_classes (produce gli score finali)
        embed_dim = self.encoder.embed_dim
        # embed_dim = 768 per ViT-base — lo leggiamo dall'encoder
        # invece di hardcodarlo per robustezza con architetture diverse

        self.classificatore = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes),
        )

        # Stampa un riepilogo del modello
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_cls = sum(p.numel() for p in self.classificatore.parameters())
        print(f"\n  MAEASTFineTune creato:")
        print(f"    Checkpoint:        {checkpoint_path}")
        print(f"    Pesi caricati:     {n_loaded}")
        print(f"    Classi output:     {num_classes}")
        print(f"    Embed dim:         {embed_dim}")
        print(f"    Param encoder:     {n_enc/1e6:.1f}M")
        print(f"    Param classif.:    {n_cls/1e6:.3f}M")
        print(f"    Param totali:      {(n_enc+n_cls)/1e6:.1f}M")

    def _carica_pesi(self, checkpoint_path: str) -> int:
        """
        Carica i pesi dal checkpoint convertito nell'encoder.

        Usa strict=False perché il checkpoint potrebbe non avere
        esattamente le stesse chiavi dell'encoder timm.
        In particolare potrebbero mancare:
          - patch_embed (se non trovato durante la conversione)
          - cls_token, pos_embed (se non presenti nel checkpoint)

        Con strict=False PyTorch carica i pesi che trova e ignora
        quelli mancanti — questi rimangono con i valori casuali
        dell'inizializzazione di timm.

        Args:
            checkpoint_path: Percorso al .pth convertito

        Returns:
            Numero di chiavi caricate con successo
        """
        # Carica il file .pth in memoria (CPU — non serve GPU)
        ck = torch.load(checkpoint_path, map_location='cpu')

        # Il checkpoint convertito ha struttura {"model": {...pesi...}}
        state_dict = ck.get('model', ck)

        # Carica i pesi nell'encoder
        # strict=False: ignora chiavi mancanti o extra invece di crashare
        esito = self.encoder.load_state_dict(state_dict, strict=False)

        n_mancanti = len(esito.missing_keys)
        n_inattesi  = len(esito.unexpected_keys)
        n_caricati  = len(state_dict) - n_inattesi

        print(f"\n  Caricamento pesi dal checkpoint:")
        print(f"    Caricati correttamente: {n_caricati}")
        print(f"    Chiavi mancanti:        {n_mancanti}")
        print(f"    Chiavi inattese:        {n_inattesi}")

        # Avviso se troppe chiavi mancano — potrebbe indicare
        # un problema nella conversione del checkpoint
        if n_mancanti > 30:
            print(f"\n  ⚠️  ATTENZIONE: {n_mancanti} chiavi mancanti!")
            print("     Potrebbe esserci un problema nel checkpoint convertito.")
            print("     Chiavi mancanti (prime 10):")
            for k in esito.missing_keys[:10]:
                print(f"       - {k}")
            print("     Suggerimento: riesegui scripts/convert_checkpoint.py")
        elif n_mancanti > 0:
            print(f"    (alcune mancanti sono normali — "
                  f"es. pos_embed ridimensionato da timm)")

        return n_caricati

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: da spettrogramma a logits di classificazione.

        Questo metodo viene chiamato automaticamente quando scrivi
        output = modello(input) — PyTorch lo chiama via __call__.

        Args:
            x: Spettrogramma mel, shape [batch, 1, 128, 512]
               batch = numero di clip nel batch (es. 24)
               1     = canali (mono, come grayscale)
               128   = frequenze mel
               512   = frame temporali

        Returns:
            logits: Tensor [batch, num_classes]
                    Uno score (non-normalizzato) per ogni classe.
                    Per ottenere probabilità: torch.softmax(logits, dim=1)
                    Per ottenere la classe predetta: torch.argmax(logits, dim=1)
        """

        # ── Encoder: spettrogramma → rappresentazione ─────────────────
        # forward_features restituisce TUTTI i token dell'encoder,
        # non solo la predizione finale.
        # Output shape: [batch, n_token + 1, embed_dim]
        #   n_token = numero di patch = (128/16) × (512/16) = 8 × 32 = 256
        #   + 1     = il CLS token speciale
        #   embed_dim = 768
        # Quindi: [batch, 257, 768]
        features = self.encoder.forward_features(x)
        # features shape: [batch, 257, 768]

        # ── Estrai il CLS token ────────────────────────────────────────
        # Il CLS token è sempre il primo token (indice 0).
        # È un token speciale che il ViT usa per riassumere
        # l'intera sequenza in un singolo vettore.
        # features[:, 0, :] → prendi TUTTI i batch, SOLO indice 0,
        #                      TUTTE le dimensioni embedding
        cls_token = features[:, 0, :]
        # cls_token shape: [batch, 768]

        # ── Classificatore: CLS token → score per classe ──────────────
        # La Sequential applica in ordine:
        # LayerNorm → Linear(768→512) → GELU → Dropout → Linear(512→50)
        logits = self.classificatore(cls_token)
        # logits shape: [batch, 50]

        return logits

    def get_optimizer_groups(
            self,
            lr_encoder: float,
            lr_classificatore: float,
            weight_decay: float = 5e-4,
    ) -> list:
        """
        Restituisce i gruppi di parametri per l'ottimizzatore,
        con learning rate diverso per encoder e classificatore.

        Perché LR diversi?
        - L'encoder ha pesi pretrainati già buoni → LR basso
          per non distruggere quello che ha imparato
        - Il classificatore parte da zero → LR più alto
          per imparare velocemente

        Esempio di utilizzo:
            gruppi = modello.get_optimizer_groups(
                lr_encoder       = 2.5e-5,
                lr_classificatore = 2.5e-4,
            )
            ottimizzatore = torch.optim.Adam(gruppi, weight_decay=5e-4)

        Args:
            lr_encoder:        Learning rate per l'encoder (es. 2.5e-5)
            lr_classificatore: Learning rate per il classificatore (es. 2.5e-4)
            weight_decay:      Regolarizzazione L2

        Returns:
            Lista di dizionari nel formato atteso da torch.optim
        """
        return [
            {
                'params':       self.encoder.parameters(),
                'lr':           lr_encoder,
                'weight_decay': weight_decay,
                'name':         'encoder',
            },
            {
                'params':       self.classificatore.parameters(),
                'lr':           lr_classificatore,
                'weight_decay': weight_decay,
                'name':         'classificatore',
            },
        ]


# ──────────────────────────────────────────────────────────────────────
# TEST — eseguibile con: python src/model.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    print("=" * 55)
    print("  TEST: src/model.py")
    print("=" * 55)

    # ── Cerca il checkpoint convertito ───────────────────────────────
    candidati = [
        "mae_ast_patch_converted.pth",
        "mae_ast_frame_converted.pth",
        "../mae_ast_patch_converted.pth",
    ]

    checkpoint = None
    for c in candidati:
        if os.path.exists(c):
            checkpoint = c
            break

    if checkpoint is None:
        print("\n  ⚠️  Nessun checkpoint convertito trovato.")
        print("  Questo è normale — il checkpoint va prima scaricato")
        print("  e convertito con scripts/convert_checkpoint.py")
        print()
        print("  Test alternativo: verifica la struttura del modello")
        print("  senza caricare pesi reali (pesi casuali)...")
        print()

        # Test senza checkpoint: crea il modello con pesi casuali
        # per verificare che l'architettura sia corretta
        class FakeEncoder(nn.Module):
            """Encoder finto per testare senza checkpoint."""
            def __init__(self):
                super().__init__()
                self.embed_dim = 768
                self._net = nn.Linear(128 * 512, 768)

            def forward_features(self, x):
                b = x.shape[0]
                # Simula output ViT: [batch, 257, 768]
                return torch.zeros(b, 257, 768)

            def parameters(self):
                return self._net.parameters()

            def load_state_dict(self, *args, **kwargs):
                from collections import namedtuple
                Result = namedtuple('Result', ['missing_keys', 'unexpected_keys'])
                return Result(missing_keys=[], unexpected_keys=[])

        # Crea modello con encoder finto
        modello = MAEASTFineTune.__new__(MAEASTFineTune)
        nn.Module.__init__(modello)
        modello.num_classes = 50
        modello.encoder = FakeEncoder()
        modello.classificatore = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 50),
        )

        print("  Modello con encoder finto creato per test struttura.")

    else:
        print(f"\n  Checkpoint trovato: {checkpoint}")
        modello = MAEASTFineTune(
            checkpoint_path = checkpoint,
            num_classes     = 50,
        )

    # ── Test forward pass ─────────────────────────────────────────────
    print("\n  Test forward pass...")
    modello.eval()

    # Crea un batch finto di 4 spettrogrammi
    x = torch.randn(4, 1, 128, 512)
    # shape: [batch=4, canali=1, freq=128, frames=512]

    with torch.no_grad():
        logits = modello(x)

    print(f"  Input shape:   {x.shape}")
    print(f"  Output shape:  {logits.shape}  (atteso: [4, 50])")
    print(f"  Output range:  [{logits.min():.3f}, {logits.max():.3f}]")

    assert logits.shape == (4, 50), \
        f"Shape attesa [4,50], ottenuta {logits.shape}"

    # ── Test predizione ────────────────────────────────────────────────
    probabilita = torch.softmax(logits, dim=1)
    classi_pred = torch.argmax(logits, dim=1)

    print(f"\n  Classi predette (random — pesi non pretrainati):")
    for i, c in enumerate(classi_pred):
        print(f"    Clip {i}: classe {c.item():2d}  "
              f"(prob: {probabilita[i, c].item():.3f})")

    # ── Test optimizer groups ─────────────────────────────────────────
    print("\n  Test get_optimizer_groups...")
    try:
        gruppi = modello.get_optimizer_groups(
            lr_encoder        = 2.5e-5,
            lr_classificatore = 2.5e-4,
        )
        print(f"  Gruppi ottimizzatore: {len(gruppi)}")
        for g in gruppi:
            n_params = sum(p.numel() for p in g['params'])
            print(f"    {g['name']:<20} lr={g['lr']:.1e}  "
                  f"params={n_params/1e6:.1f}M")
    except AttributeError:
        print("  (get_optimizer_groups non disponibile con encoder finto)")

    print(f"\n  ✅ Tutti i controlli superati!")
    print("=" * 55)