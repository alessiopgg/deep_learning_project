"""
scripts/convert_checkpoint.py
==============================
Converte un checkpoint MAE-AST da formato fairseq a formato timm.

PERCHÉ SERVE:
    Il checkpoint degli autori è salvato con fairseq (framework di
    Facebook). Il nostro codice di fine-tuning usa timm (PyTorch
    Image Models). Le due librerie usano nomi diversi per gli stessi
    pesi — questo script fa la traduzione.

IL PROBLEMA PRINCIPALE — Q, K, V:
    Fairseq salva le proiezioni attention in tre matrici separate:
        encoder.layers.0.self_attn.q_proj.weight  [768, 768]
        encoder.layers.0.self_attn.k_proj.weight  [768, 768]
        encoder.layers.0.self_attn.v_proj.weight  [768, 768]

    timm le vuole concatenate in una sola matrice QKV:
        blocks.0.attn.qkv.weight                  [2304, 768]
        (prime 768 righe = Q, poi K, poi V)

    Questo script fa torch.cat([Q, K, V], dim=0) per ogni layer.

MAPPING COMPLETO DELLE CHIAVI:
    fairseq                              timm
    ─────────────────────────────────    ──────────────────────────
    encoder.layers.N.self_attn_layer_norm  →  blocks.N.norm1
    encoder.layers.N.self_attn.q_proj   )
    encoder.layers.N.self_attn.k_proj   )→  blocks.N.attn.qkv
    encoder.layers.N.self_attn.v_proj   )
    encoder.layers.N.self_attn.out_proj  →  blocks.N.attn.proj
    encoder.layers.N.final_layer_norm    →  blocks.N.norm2
    encoder.layers.N.fc1                 →  blocks.N.mlp.fc1
    encoder.layers.N.fc2                 →  blocks.N.mlp.fc2
    encoder.layer_norm                   →  norm
    encoder.embed_tokens / patch_embed   →  patch_embed.proj

USO:
    python scripts/convert_checkpoint.py <input.pt> <output.pth>

ESEMPIO:
    python scripts/convert_checkpoint.py \\
        chunk_patch_75_12LayerEncoder.pt \\
        mae_ast_patch_converted.pth
"""

import sys
import os
import torch

# ──────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ──────────────────────────────────────────────────────────────────────

# Numero di layer dell'encoder nel checkpoint (12 per il modello base)
NUM_LAYERS = 12

# Dimensione dell'embedding (768 per ViT-base)
# Lo script prova a rilevarlo automaticamente dal checkpoint,
# ma questo valore è usato come fallback
EMBED_DIM_DEFAULT = 768


# ──────────────────────────────────────────────────────────────────────
# FUNZIONI
# ──────────────────────────────────────────────────────────────────────

def estrai_state_dict(checkpoint: dict) -> dict:
    """
    Estrae il sotto-dizionario con i pesi del modello.

    Un checkpoint fairseq ha questa struttura:
    {
        'model':            { ...pesi... },   ← quello che vogliamo
        'args':             Namespace(...),
        'extra_state':      {...},
        'optimizer_history': [...],
        'last_optimizer_state': {...},
    }

    Args:
        checkpoint: Il dizionario caricato da torch.load()

    Returns:
        Il dizionario con i pesi (chiave → tensore)
    """
    if "model" in checkpoint:
        print("    ✅ Pesi trovati in checkpoint['model']")
        return checkpoint["model"]

    if "state_dict" in checkpoint:
        print("    ✅ Pesi trovati in checkpoint['state_dict']")
        return checkpoint["state_dict"]

    # Alcuni checkpoint hanno i pesi direttamente al livello radice
    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        print("    ✅ Il checkpoint stesso è il dizionario pesi")
        return checkpoint

    raise ValueError(
        "Struttura del checkpoint non riconosciuta.\n"
        "Esegui prima inspect_checkpoint.py per vedere la struttura."
    )


def rileva_embed_dim(state_dict: dict, default: int) -> int:
    """
    Tenta di rilevare automaticamente la dimensione embedding
    leggendo la shape del primo tensore Q trovato.

    La dimensione embedding è la larghezza di ogni layer transformer.
    Per ViT-base è 768, per ViT-small è 384, per ViT-tiny è 192.

    Args:
        state_dict: Dizionario dei pesi
        default:    Valore da usare se non riesce a rilevarlo

    Returns:
        La dimensione embedding rilevata o il valore di default
    """
    for chiave, tensore in state_dict.items():
        if "q_proj.weight" in chiave and isinstance(tensore, torch.Tensor):
            dim = tensore.shape[0]
            print(f"    ✅ Embed dim rilevata automaticamente: {dim}")
            return dim
        if "attn.qkv.weight" in chiave and isinstance(tensore, torch.Tensor):
            dim = tensore.shape[0] // 3
            print(f"    ✅ Embed dim rilevata da qkv: {dim}")
            return dim

    print(f"    ⚠️  Embed dim non rilevata, uso default: {default}")
    return default


def rileva_prefisso(state_dict: dict) -> str:
    """
    Rileva il prefisso usato per le chiavi dell'encoder.

    In fairseq il prefisso più comune è "encoder." ma alcuni
    checkpoint usano varianti diverse.

    Esempi di chiavi con prefisso:
        "encoder.layers.0.self_attn.q_proj.weight"
    Esempi di chiavi senza prefisso:
        "layers.0.self_attn.q_proj.weight"

    Returns:
        Il prefisso rilevato (stringa, può essere vuota "")
    """
    for k in state_dict.keys():
        if "layers.0.self_attn" in k:
            if k.startswith("encoder."):
                print("    ✅ Prefisso rilevato: 'encoder.'")
                return "encoder."
            else:
                print("    ✅ Prefisso rilevato: '' (vuoto)")
                return ""

    print("    ⚠️  Prefisso non rilevato, provo 'encoder.'")
    return "encoder."


def converti_layer(
        state_in: dict,
        state_out: dict,
        n: int,
        prefisso: str,
) -> bool:
    """
    Converte tutti i pesi di UN singolo layer transformer.

    Questa funzione viene chiamata 12 volte (una per layer).
    Gestisce il mapping completo:
        - Layer norm pre-attention  (norm1)
        - Q + K + V → QKV concatenato
        - Proiezione output attention (attn.proj)
        - Layer norm pre-feedforward (norm2)
        - Feed-forward fc1 e fc2

    Args:
        state_in:  Dizionario pesi input (formato fairseq)
        state_out: Dizionario pesi output (formato timm) — modificato in-place
        n:         Indice del layer (0, 1, ..., 11)
        prefisso:  Prefisso da usare nelle chiavi input

    Returns:
        True se il layer è stato convertito, False se non trovato
    """
    # Prefissi per le chiavi di input (fairseq) e output (timm)
    p_in = f"{prefisso}layers.{n}."
    p_out = f"blocks.{n}."

    # Controlla che questo layer esista nel checkpoint
    chiave_test = f"{p_in}self_attn_layer_norm.weight"
    chiave_alt = f"{p_in}self_attn.in_proj_weight"
    if chiave_test not in state_in and chiave_alt not in state_in:
        return False

    # ── 1. Layer norm pre-attention: norm1 ───────────────────────────
    # fairseq: self_attn_layer_norm.{weight,bias}
    # timm:    norm1.{weight,bias}
    for s in ["weight", "bias"]:
        k_in = f"{p_in}self_attn_layer_norm.{s}"
        k_out = f"{p_out}norm1.{s}"
        if k_in in state_in:
            state_out[k_out] = state_in[k_in].clone()

    # ── 2. Attention Q, K, V → QKV concatenato ───────────────────────
    # Caso A: fairseq salva Q, K, V come tre matrici separate
    #   q_proj.weight [768, 768]
    #   k_proj.weight [768, 768]
    #   v_proj.weight [768, 768]
    # → concat lungo dim=0 → [2304, 768]
    #
    # Caso B: fairseq salva già tutto in in_proj_weight [2304, 768]
    # (meno comune, ma possibile)

    q_w = f"{p_in}self_attn.q_proj.weight"
    k_w = f"{p_in}self_attn.k_proj.weight"
    v_w = f"{p_in}self_attn.v_proj.weight"
    q_b = f"{p_in}self_attn.q_proj.bias"
    k_b = f"{p_in}self_attn.k_proj.bias"
    v_b = f"{p_in}self_attn.v_proj.bias"
    in_w = f"{p_in}self_attn.in_proj_weight"
    in_b = f"{p_in}self_attn.in_proj_bias"

    if all(k in state_in for k in [q_w, k_w, v_w]):
        # Caso A: concatena Q, K, V
        # torch.cat([Q, K, V], dim=0) impila le righe:
        #   Q: righe 0-767
        #   K: righe 768-1535
        #   V: righe 1536-2303
        qkv_weight = torch.cat([
            state_in[q_w],  # [768, 768]
            state_in[k_w],  # [768, 768]
            state_in[v_w],  # [768, 768]
        ], dim=0)  # → [2304, 768]
        state_out[f"{p_out}attn.qkv.weight"] = qkv_weight

    elif in_w in state_in:
        # Caso B: già concatenato, copia direttamente
        state_out[f"{p_out}attn.qkv.weight"] = state_in[in_w].clone()

    # Stesso per i bias
    if all(k in state_in for k in [q_b, k_b, v_b]):
        qkv_bias = torch.cat([
            state_in[q_b],  # [768]
            state_in[k_b],  # [768]
            state_in[v_b],  # [768]
        ], dim=0)  # → [2304]
        state_out[f"{p_out}attn.qkv.bias"] = qkv_bias

    elif in_b in state_in:
        state_out[f"{p_out}attn.qkv.bias"] = state_in[in_b].clone()

    # ── 3. Proiezione output attention ────────────────────────────────
    # fairseq: self_attn.out_proj.{weight,bias}
    # timm:    attn.proj.{weight,bias}
    for s in ["weight", "bias"]:
        k_in = f"{p_in}self_attn.out_proj.{s}"
        k_out = f"{p_out}attn.proj.{s}"
        if k_in in state_in:
            state_out[k_out] = state_in[k_in].clone()

    # ── 4. Layer norm pre-feedforward: norm2 ─────────────────────────
    # fairseq: final_layer_norm.{weight,bias}
    # timm:    norm2.{weight,bias}
    for s in ["weight", "bias"]:
        k_in = f"{p_in}final_layer_norm.{s}"
        k_out = f"{p_out}norm2.{s}"
        if k_in in state_in:
            state_out[k_out] = state_in[k_in].clone()

    # ── 5. Feed-forward (MLP): fc1 e fc2 ─────────────────────────────
    # fairseq: fc1.{weight,bias}, fc2.{weight,bias}
    # timm:    mlp.fc1.{weight,bias}, mlp.fc2.{weight,bias}
    for fc in ["fc1", "fc2"]:
        for s in ["weight", "bias"]:
            k_in = f"{p_in}{fc}.{s}"
            k_out = f"{p_out}mlp.{fc}.{s}"
            if k_in in state_in:
                state_out[k_out] = state_in[k_in].clone()

    return True


def converti(input_path: str, output_path: str) -> None:
    """
    Funzione principale: esegue l'intera conversione.

    Pipeline:
        1. Carica il checkpoint fairseq
        2. Estrae i pesi del modello
        3. Rileva prefisso ed embed_dim automaticamente
        4. Converte tutti i layer 0-11
        5. Converte la layer norm finale
        6. Tenta di copiare patch embedding e pos embedding
        7. Salva il nuovo checkpoint in formato timm
        8. Verifica il file salvato

    Args:
        input_path:  Percorso al .pt fairseq
        output_path: Percorso dove salvare il .pth timm
    """

    print(f"\n{'=' * 60}")
    print("  CONVERSIONE CHECKPOINT  fairseq → timm")
    print(f"  Input:  {os.path.basename(input_path)}")
    print(f"  Output: {os.path.basename(output_path)}")
    print(f"{'=' * 60}")

    # ── Step 1: Caricamento ───────────────────────────────────────────
    if not os.path.exists(input_path):
        print(f"\n❌  File non trovato: {input_path}")
        sys.exit(1)

    dim_mb = os.path.getsize(input_path) / 1e6
    print(f"\n[1/6] Caricamento ({dim_mb:.0f} MB)...")
    ck = torch.load(input_path, map_location="cpu")
    print("      ✅ Caricato")

    # ── Step 2: Estrazione pesi ───────────────────────────────────────
    print("\n[2/6] Estrazione pesi...")
    state_in = estrai_state_dict(ck)
    print(f"      Tensori trovati: {len(state_in)}")

    # ── Step 3: Rilevamento automatico parametri ──────────────────────
    print("\n[3/6] Rilevamento architettura...")
    embed_dim = rileva_embed_dim(state_in, EMBED_DIM_DEFAULT)
    prefisso = rileva_prefisso(state_in)

    # ── Step 4: Conversione layer per layer ───────────────────────────
    print(f"\n[4/6] Conversione {NUM_LAYERS} layer encoder...")
    state_out = {}
    layer_ok = 0

    for n in range(NUM_LAYERS):
        ok = converti_layer(state_in, state_out, n, prefisso)
        if ok:
            layer_ok += 1
            print(f"      Layer {n:2d}  ✅  "
                  f"({sum(1 for k in state_out if f'blocks.{n}.' in k)} chiavi)")
        else:
            print(f"      Layer {n:2d}  ⚠️   non trovato")

    # ── Step 5: Layer norm finale ─────────────────────────────────────
    # fairseq: encoder.layer_norm  oppure  layer_norm
    # timm:    norm
    print("\n[5/6] Layer norm finale e componenti globali...")

    for candidato in [
        f"{prefisso}layer_norm.weight",
        "layer_norm.weight",
    ]:
        if candidato in state_in:
            state_out["norm.weight"] = state_in[candidato].clone()
            print(f"      norm.weight  ✅  (da '{candidato}')")
            break
    else:
        print("      norm.weight  ⚠️   non trovato")

    for candidato in [
        f"{prefisso}layer_norm.bias",
        "layer_norm.bias",
    ]:
        if candidato in state_in:
            state_out["norm.bias"] = state_in[candidato].clone()
            print(f"      norm.bias    ✅  (da '{candidato}')")
            break

    # ── post_extract_proj → patch_embed ───────────────────────────────
    # ── post_extract_proj → patch_embed ───────────────────────────────
    # QUESTO E' IL FIX CRITICO per la Parte B.
    #
    # Il problema: fairseq e timm rappresentano lo stesso strato
    # (trasformazione patch → vettore) con nomi e shape diversi.
    #
    # fairseq: post_extract_proj.weight  [768, 256]
    #   - matrice lineare 2D
    #   - 256 = valori di una patch 16×16
    #
    # timm:    patch_embed.proj.weight   [768, 1, 16, 16]
    #   - convoluzione 4D
    #   - equivalente matematicamente: 256 = 1 × 16 × 16
    #
    # Il reshape [768, 256] → [768, 1, 16, 16] e esatto
    # perché le due operazioni producono lo stesso risultato
    # su qualsiasi input.
    k_w = 'post_extract_proj.weight'
    k_b = 'post_extract_proj.bias'

    if k_w in state_in and 'patch_embed.proj.weight' not in state_out:
        w = state_in[k_w]              # [768, 256]
        w = w.reshape(768, 1, 16, 16)  # [768, 1, 16, 16]
        state_out['patch_embed.proj.weight'] = w
        print(f"      patch_embed.proj.weight  ✅  "
              f"(reshapato da post_extract_proj "
              f"[768,256] → [768,1,16,16])")

    if k_b in state_in and 'patch_embed.proj.bias' not in state_out:
        state_out['patch_embed.proj.bias'] = state_in[k_b].clone()
        print(f"      patch_embed.proj.bias    ✅  "
              f"(da post_extract_proj)")

    # ── Patch embedding ───────────────────────────────────────────────
    # La convoluzione che trasforma le patch in vettori. 
    # I nomi variano tra versioni di fairseq — proviamo vari candidati.
    candidati_patch = [
        (f"{prefisso}patch_embed.proj.weight", "patch_embed.proj.weight"),
        (f"{prefisso}patch_embed.proj.bias", "patch_embed.proj.bias"),
        ("ast_patch_embed.proj.weight", "patch_embed.proj.weight"),
        ("ast_patch_embed.proj.bias", "patch_embed.proj.bias"),
        ("patch_embed.proj.weight", "patch_embed.proj.weight"),
        ("patch_embed.proj.bias", "patch_embed.proj.bias"),
    ]
    for k_in, k_out in candidati_patch:
        if k_in in state_in and k_out not in state_out:
            state_out[k_out] = state_in[k_in].clone()
            print(f"      {k_out:<35} ✅  (da '{k_in}')")

    # ── Cls token e positional embedding ─────────────────────────────
    for k_in, k_out in [
        (f"{prefisso}cls_token", "cls_token"),
        ("cls_token", "cls_token"),
        (f"{prefisso}pos_embed", "pos_embed"),
        ("pos_embed", "pos_embed"),
    ]:
        if k_in in state_in and k_out not in state_out:
            state_out[k_out] = state_in[k_in].clone()
            print(f"      {k_out:<35} ✅")

    # ── Step 6: Salvataggio ───────────────────────────────────────────
    print(f"\n[6/6] Salvataggio → {output_path}")
    # Salviamo con la chiave 'model' — formato standard per timm/SSAST
    torch.save({"model": state_out}, output_path)
    dim_out = os.path.getsize(output_path) / 1e6
    print(f"      ✅ Salvato ({dim_out:.0f} MB)")

    # ── Report finale ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  REPORT")
    print(f"{'=' * 60}")
    print(f"  Layer convertiti:        {layer_ok}/{NUM_LAYERS}")
    print(f"  Chiavi nell'output:      {len(state_out)}")

    # Chiavi input non mappate (informativo)
    chiavi_mappate = set()
    for n in range(NUM_LAYERS):
        p = f"{prefisso}layers.{n}."
        for k in state_in:
            if k.startswith(p):
                chiavi_mappate.add(k)
    non_mappate = [k for k in state_in if k not in chiavi_mappate]

    encoder_non_mappate = [k for k in non_mappate
                           if not k.startswith(f"{prefisso}decoder")
                           and "decoder" not in k]

    print(f"  Chiavi non mappate:      {len(non_mappate)}")
    print(f"    di cui decoder:        "
          f"{len(non_mappate) - len(encoder_non_mappate)}  "
          f"(normale — non servono per fine-tuning)")
    print(f"    di cui non-decoder:    {len(encoder_non_mappate)}")

    if encoder_non_mappate:
        print("\n  Chiavi non-decoder non mappate (da controllare):")
        for k in sorted(encoder_non_mappate)[:15]:
            v = state_in[k]
            shape = list(v.shape) if isinstance(v, torch.Tensor) else "non-Tensor"
            print(f"    ⚪ {k}  {shape}")
        if len(encoder_non_mappate) > 15:
            print(f"    ... e altri {len(encoder_non_mappate) - 15}")

    # ── Verifica ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  VERIFICA")
    print(f"{'=' * 60}")
    verifica(output_path)


def verifica(output_path: str) -> None:
    """
    Ricarica il checkpoint salvato e controlla che sia integro.

    Verifica la presenza delle chiavi fondamentali che timm
    si aspetta trovare quando carica un ViT-base.
    """
    try:
        ck = torch.load(output_path, map_location="cpu")
        state = ck.get("model", ck)

        # Chiavi obbligatorie per un ViT-base funzionante
        controlli = {
            "blocks.0.attn.qkv.weight": "QKV layer 0",
            "blocks.11.attn.qkv.weight": "QKV layer 11",
            "blocks.0.norm1.weight": "norm1 layer 0",
            "blocks.0.mlp.fc1.weight": "MLP fc1 layer 0",
            "norm.weight": "layer norm finale",
        }

        tutti_ok = True
        for chiave, descrizione in controlli.items():
            presente = chiave in state
            icona = "✅" if presente else "❌"
            print(f"  {icona}  {descrizione:<25} ({chiave})")
            if not presente:
                tutti_ok = False

        # Conta chiavi opzionali
        ha_patch = "patch_embed.proj.weight" in state
        ha_cls = "cls_token" in state
        ha_pos = "pos_embed" in state
        print(f"\n  {'✅' if ha_patch else '⚠️ '} patch_embed.proj.weight")
        print(f"  {'✅' if ha_cls else '⚠️ '} cls_token")
        print(f"  {'✅' if ha_pos else '⚠️ '} pos_embed")

        print()
        if tutti_ok:
            print("  ✅ Il checkpoint è pronto per il fine-tuning!")
            print(f"\n  Prossimo step: carica questo file su Google Drive")
            print(f"  e usalo nel notebook PartB_finetune_ESC50.ipynb")
        else:
            print("  ❌ Alcune chiavi obbligatorie mancano.")
            print("     Esegui inspect_checkpoint.py sull'input")
            print("     per capire la struttura delle chiavi.")

    except Exception as e:
        print(f"  ❌ Errore durante la verifica: {e}")

    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUso:")
        print("  python scripts/convert_checkpoint.py "
              "<input.pt> <output.pth>")
        print()
        print("Esempio (modello Patch):")
        print("  python scripts/convert_checkpoint.py \\")
        print("    chunk_patch_75_12LayerEncoder.pt \\")
        print("    mae_ast_patch_converted.pth")
        print()
        print("Esempio (modello Frame):")
        print("  python scripts/convert_checkpoint.py \\")
        print("    random_frame_75_12LayerEncoder.pt \\")
        print("    mae_ast_frame_converted.pth")
        print()
        sys.exit(0)

    converti(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
    )
