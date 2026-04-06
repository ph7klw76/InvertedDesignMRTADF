#!/usr/bin/env python3
"""
e2e_pipeline.py — End-to-end MR-TADF molecular generation pipeline
====================================================================

Integrates all four components into a single workflow:

  1. QM9 pre-training + domain adaptation  (transfer_learning.py)
  2. GNN property predictor               (gnn_predictor.py)
  3. SELFIES conditional VAE generation    (selfies_generator.py)
  4. Novelty & chemical-plausibility gate  (novelty_validation.py)

Workflow:
  ┌──────────────────────────────────────────────────┐
  │ Phase A: Pre-train backbone on QM9 descriptors   │
  │ Phase B: MMD domain adaptation → MR-TADF         │
  │ Phase C: Progressive fine-tune on MR-TADF        │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Train GNN predictor on MR-TADF molecular graphs  │
  │ (operates on SMILES → graph, independent of      │
  │  descriptor pipeline; provides ensemble scoring)  │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Train SELFIES cVAE on MR-TADF SMILES + targets   │
  │ → generates novel valid molecules directly        │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Conditional generation:                           │
  │   Set target y* = [T1-S1=0.04, T2-S1=0.05, ...]  │
  │   Sample z ~ N(0,I), decode to SELFIES → SMILES   │
  │   Score with GNN + transfer-learned predictor      │
  │   Filter: property compliance + novelty + chem.    │
  │   Rank and export top candidates                   │
  └──────────────────────────────────────────────────┘

Usage:
  python e2e_pipeline.py \\
    --descriptor_file <path.xlsx> \\
    --target_file <path.xlsx> \\
    --qm9_file <qm9_descriptors.csv>  \\  # optional
    --device cuda \\
    --n_generate 1000

Dependencies:
  pip install torch torch-geometric selfies rdkit-pypi mordred \\
              pandas numpy scikit-learn openpyxl
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

E2E_CONFIG = {
    # Transfer learning
    "tl_pretrain_epochs": 100,
    "tl_mmd_epochs": 50,
    "tl_finetune_epochs": 300,
    "tl_hidden_dim": 256,
    "tl_n_blocks": 4,

    # GNN
    "gnn_hidden_dim": 256,
    "gnn_n_layers": 6,
    "gnn_epochs": 300,
    "gnn_lr": 5e-4,

    # SELFIES VAE
    "selfies_d_model": 256,
    "selfies_n_heads": 8,
    "selfies_enc_layers": 4,
    "selfies_dec_layers": 6,
    "selfies_latent_dim": 128,
    "selfies_epochs": 200,
    "selfies_lr": 3e-4,
    "selfies_beta_max": 0.5,

    # Generation
    "n_generate": 1000,
    "n_samples_per_target": 10,
    "temperature": 0.8,
    "top_k": 0,
    "t1s1_target": 0.04,
    "t2s1_target": 0.05,

    # Screening
    "tanimoto_threshold": 0.85,
    "ensemble_weight_gnn": 0.5,

    # General
    "batch_size": 32,
    "patience": 30,
    "mi_top_k": 200,
}


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_e2e_pipeline(
    descriptor_file: str,
    target_file: str,
    qm9_file: Optional[str] = None,
    device: str = "cpu",
    config: Optional[Dict] = None,
) -> Dict:
    """
    Execute the complete end-to-end pipeline.

    Args:
        descriptor_file: path to MR-TADF descriptor Excel file
        target_file: path to MR-TADF target properties Excel file
        qm9_file: path to QM9 descriptors CSV (optional; skip pre-training if None)
        device: 'cpu' or 'cuda'
        config: override configuration dict

    Returns:
        dict with all results, trained models, and generated candidates
    """
    cfg = {**E2E_CONFIG, **(config or {})}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"e2e_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ────────────────────────────────────────────────────────────
    #  Stage 1: Data Preparation
    # ────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STAGE 1: Data Preparation")
    logger.info("=" * 70)

    from data_processing import prepare_dataset, TARGET_COLS

    data = prepare_dataset(descriptor_file, target_file, mi_top_k=cfg["mi_top_k"])
    input_dim = data["X_train"].shape[1]

    logger.info(f"  Features: {input_dim}, Train: {len(data['X_train'])}, Val: {len(data['X_val'])}")

    results = {"config": cfg, "data": data}

    # ────────────────────────────────────────────────────────────
    #  Stage 2: Transfer Learning (QM9 → MR-TADF)
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Transfer Learning from QM9")
    logger.info("=" * 70)

    from transfer_learning import (
        TransferableBackbone, TransferLearningTrainer,
        run_transfer_learning_pipeline, QM9DescriptorComputer,
    )

    backbone = TransferableBackbone(
        input_dim=input_dim,
        hidden_dim=cfg["tl_hidden_dim"],
        n_blocks=cfg["tl_n_blocks"],
    )

    if qm9_file is not None:
        # Load QM9 descriptors
        logger.info(f"  Loading QM9 from {qm9_file}")
        qm9_df = pd.read_csv(qm9_file)

        # Assume CSV has feature columns matching MR-TADF + 5 property columns
        qm9_feat_cols = [c for c in data["feature_names"] if c in qm9_df.columns]
        qm9_X = qm9_df[qm9_feat_cols].values.astype(np.float64)
        qm9_X = np.nan_to_num(qm9_X)

        # Scale using the same scaler as MR-TADF
        qm9_X_scaled = data["scaler_X"].transform(qm9_X)

        # QM9 properties: HOMO, LUMO, gap, dipole, polarisability
        qm9_prop_cols = ["HOMO", "LUMO", "gap", "dipole", "alpha"]
        qm9_y = qm9_df[qm9_prop_cols].values.astype(np.float64)
        from sklearn.preprocessing import StandardScaler
        qm9_y_scaler = StandardScaler().fit(qm9_y)
        qm9_y_scaled = qm9_y_scaler.transform(qm9_y)

        backbone, tadf_head, tl_metrics = run_transfer_learning_pipeline(
            qm9_descriptors=qm9_X_scaled,
            qm9_properties=qm9_y_scaled,
            tadf_X_train=data["X_train"],
            tadf_y_train=data["y_train"],
            tadf_X_val=data["X_val"],
            tadf_y_val=data["y_val"],
            input_dim=input_dim,
            scaler_y=data["scaler_y"],
            device=device,
        )

        results["transfer_learning"] = tl_metrics
        torch.save(backbone.state_dict(), output_dir / "backbone_pretrained.pt")
        torch.save(tadf_head.state_dict(), output_dir / "tadf_head.pt")

    else:
        logger.info("  No QM9 file provided — training from scratch with transfer architecture")
        trainer = TransferLearningTrainer(backbone=backbone, device=device)
        tadf_head, ft_metrics = trainer.finetune_on_tadf(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            max_epochs=cfg["tl_finetune_epochs"],
            patience=cfg["patience"],
            scaler_y=data["scaler_y"],
        )
        results["transfer_learning"] = {"finetune": ft_metrics}
        torch.save(backbone.state_dict(), output_dir / "backbone.pt")
        torch.save(tadf_head.state_dict(), output_dir / "tadf_head.pt")

    # ────────────────────────────────────────────────────────────
    #  Stage 3: GNN Property Predictor
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: GNN Property Predictor")
    logger.info("=" * 70)

    try:
        from gnn_predictor import (
            GNNPropertyPredictor, GNNTrainer, MolecularGraphBuilder,
        )

        # Build molecular graphs
        all_smiles = list(data["smiles_train"]) + list(data["smiles_val"])
        all_y = np.vstack([data["y_train"], data["y_val"]])

        graphs, valid_idx = MolecularGraphBuilder.batch_smiles_to_graphs(
            all_smiles, all_y
        )

        if len(graphs) >= 10:
            # Split into train/val (respecting original split)
            n_train = len(data["smiles_train"])
            train_graphs = [g for i, g in enumerate(graphs) if valid_idx[i] < n_train]
            val_graphs = [g for i, g in enumerate(graphs) if valid_idx[i] >= n_train]

            gnn = GNNPropertyPredictor(
                hidden_dim=cfg["gnn_hidden_dim"],
                n_conv_layers=cfg["gnn_n_layers"],
            )

            gnn_trainer = GNNTrainer(
                model=gnn, lr=cfg["gnn_lr"],
                max_epochs=cfg["gnn_epochs"],
                patience=cfg["patience"],
                batch_size=cfg["batch_size"],
                device=device,
                scaler_y=data["scaler_y"],
            )

            gnn_metrics = gnn_trainer.fit(train_graphs, val_graphs)
            results["gnn"] = gnn_metrics
            torch.save(gnn.state_dict(), output_dir / "gnn.pt")
            logger.info(f"  GNN: MAE_T1S1={gnn_metrics.get('mae_T1-S1', 'N/A'):.4f}")
        else:
            logger.warning("  Too few valid graphs for GNN training")
            gnn = None
            results["gnn"] = {"error": "insufficient valid graphs"}

    except ImportError as e:
        logger.warning(f"  GNN skipped (missing dependency): {e}")
        gnn = None
        results["gnn"] = {"error": str(e)}

    # ────────────────────────────────────────────────────────────
    #  Stage 4: SELFIES Conditional VAE
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: SELFIES Conditional VAE Training")
    logger.info("=" * 70)

    try:
        from selfies_generator import (
            SELFIESTokeniser, SELFIESConditionalVAE, SELFIESVAETrainer,
        )

        # Build SELFIES tokeniser from all training SMILES
        tokeniser = SELFIESTokeniser(max_length=256)
        tokeniser.fit(list(data["smiles_train"]) + list(data["smiles_val"]))

        selfies_vae = SELFIESConditionalVAE(
            vocab_size=tokeniser.vocab_size,
            max_length=256,
            d_model=cfg["selfies_d_model"],
            n_heads=cfg["selfies_n_heads"],
            n_encoder_layers=cfg["selfies_enc_layers"],
            n_decoder_layers=cfg["selfies_dec_layers"],
            latent_dim=cfg["selfies_latent_dim"],
            property_dim=len(TARGET_COLS),
            dropout=0.1,
        )

        vae_trainer = SELFIESVAETrainer(
            model=selfies_vae,
            tokeniser=tokeniser,
            lr=cfg["selfies_lr"],
            beta_max=cfg["selfies_beta_max"],
            max_epochs=cfg["selfies_epochs"],
            patience=cfg["patience"],
            batch_size=cfg["batch_size"],
            device=device,
        )

        vae_metrics = vae_trainer.fit(
            list(data["smiles_train"]), data["y_train"],
            list(data["smiles_val"]), data["y_val"],
        )

        results["selfies_vae"] = vae_metrics
        torch.save(selfies_vae.state_dict(), output_dir / "selfies_vae.pt")

    except ImportError as e:
        logger.warning(f"  SELFIES VAE skipped (missing dependency): {e}")
        selfies_vae = None
        vae_trainer = None
        tokeniser = None
        results["selfies_vae"] = {"error": str(e)}

    # ────────────────────────────────────────────────────────────
    #  Stage 5: Conditional Molecular Generation
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5: Conditional Molecular Generation")
    logger.info("=" * 70)

    generated_smiles = []
    generation_stats = {}

    if selfies_vae is not None and vae_trainer is not None:
        # Construct target property vectors (in scaled space)
        target_raw = np.array([[
            cfg["t1s1_target"],    # T1-S1
            cfg["t2s1_target"],    # T2-S1
            0.06,                  # DeltaEST
            2.85,                  # S1 energy
            2.80,                  # T1 energy
            0.70,                  # oscillator strength
        ]])
        target_scaled = data["scaler_y"].transform(target_raw)

        # Generate variations of targets
        n_target_variations = cfg["n_generate"] // cfg["n_samples_per_target"]
        target_variations = []
        rng = np.random.RandomState(42)
        for _ in range(n_target_variations):
            noise = rng.normal(0, 0.1, size=target_scaled.shape)
            # Keep T1-S1 and T2-S1 very tight
            noise[0, 0] *= 0.3  # less noise on T1-S1
            noise[0, 1] *= 0.3  # less noise on T2-S1
            target_variations.append(target_scaled + noise)

        target_batch = np.vstack(target_variations)

        logger.info(f"  Generating {cfg['n_generate']} molecules from {n_target_variations} target variations")

        generated_smiles = vae_trainer.generate_molecules(
            target_properties=target_batch,
            n_samples=cfg["n_samples_per_target"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
        )

        n_valid = sum(1 for s in generated_smiles if s is not None)
        generation_stats["n_generated"] = len(generated_smiles)
        generation_stats["n_valid_smiles"] = n_valid
        generation_stats["validity_rate"] = n_valid / max(len(generated_smiles), 1)

        logger.info(f"  Validity: {n_valid}/{len(generated_smiles)} ({generation_stats['validity_rate']*100:.1f}%)")

    results["generation"] = generation_stats

    # ────────────────────────────────────────────────────────────
    #  Stage 6: Multi-Model Screening
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 6: Multi-Model Screening & Novelty Validation")
    logger.info("=" * 70)

    valid_smiles = [s for s in generated_smiles if s is not None]

    if len(valid_smiles) > 0:
        from novelty_validation import (
            NoveltyValidator, MRTADFChemicalFilter,
        )

        # 6a: Chemical plausibility filter (MR-TADF specific)
        chemically_valid = []
        for smi in valid_smiles:
            passes, details = MRTADFChemicalFilter.filter_molecule(smi)
            if passes:
                chemically_valid.append(smi)

        logger.info(f"  Chemical filter: {len(chemically_valid)}/{len(valid_smiles)} pass")

        # 6b: Novelty validation
        known_smiles = list(data["smiles_train"]) + list(data["smiles_val"])
        novelty_validator = NoveltyValidator(
            known_smiles, tanimoto_threshold=cfg["tanimoto_threshold"]
        )

        novel_molecules = novelty_validator.filter_novel(
            chemically_valid, require_level=2
        )
        logger.info(f"  Novelty filter: {len(novel_molecules)}/{len(chemically_valid)} pass")

        # 6c: GNN property prediction for scoring (if available)
        scored_candidates = []

        if gnn is not None and len(novel_molecules) > 0:
            try:
                gnn.eval()
                for mol_info in novel_molecules:
                    smi = mol_info.get("canonical_smiles", mol_info.get("smiles"))
                    graph = MolecularGraphBuilder.smiles_to_graph(smi)
                    if graph is None:
                        continue

                    from torch_geometric.data import Batch
                    batch = Batch.from_data_list([graph]).to(device)
                    with torch.no_grad():
                        pred = gnn.forward_from_data(batch).cpu().numpy()

                    pred_raw = data["scaler_y"].inverse_transform(pred)[0]

                    scored_candidates.append({
                        "smiles": smi,
                        "scaffold": mol_info.get("scaffold"),
                        "max_tanimoto": mol_info.get("max_tanimoto", 0),
                        "T1_S1_gnn": float(pred_raw[0]),
                        "T2_S1_gnn": float(pred_raw[1]),
                        "DeltaEST_gnn": float(pred_raw[2]),
                        "S1_gnn": float(pred_raw[3]),
                        "T1_gnn": float(pred_raw[4]),
                        "f_gnn": float(pred_raw[5]),
                    })
            except Exception as e:
                logger.warning(f"  GNN scoring failed: {e}")
                # Fall back to non-scored
                scored_candidates = [
                    {"smiles": m.get("canonical_smiles", m.get("smiles")),
                     "max_tanimoto": m.get("max_tanimoto", 0)}
                    for m in novel_molecules
                ]
        else:
            scored_candidates = [
                {"smiles": m.get("canonical_smiles", m.get("smiles")),
                 "max_tanimoto": m.get("max_tanimoto", 0)}
                for m in novel_molecules
            ]

        # 6d: Rank by predicted T1-S1 + T2-S1 (lower = better)
        if scored_candidates and "T1_S1_gnn" in scored_candidates[0]:
            scored_candidates.sort(
                key=lambda x: abs(x.get("T1_S1_gnn", 999)) + abs(x.get("T2_S1_gnn", 999))
            )
            # Filter by property thresholds
            final_candidates = [
                c for c in scored_candidates
                if abs(c.get("T1_S1_gnn", 999)) < 0.08
                and abs(c.get("T2_S1_gnn", 999)) < 0.10
            ]
        else:
            final_candidates = scored_candidates

        logger.info(f"  Final candidates meeting all criteria: {len(final_candidates)}")

        results["screening"] = {
            "n_valid_smiles": len(valid_smiles),
            "n_chemically_valid": len(chemically_valid),
            "n_novel": len(novel_molecules),
            "n_final_candidates": len(final_candidates),
        }

    else:
        final_candidates = []
        results["screening"] = {"error": "No valid SMILES generated"}

    # ────────────────────────────────────────────────────────────
    #  Stage 7: Export Results
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 7: Export")
    logger.info("=" * 70)

    # Export top candidates
    top_n = min(50, len(final_candidates))
    if top_n > 0:
        df_candidates = pd.DataFrame(final_candidates[:top_n])
        df_candidates.insert(0, "rank", range(1, top_n + 1))
        df_candidates.to_csv(output_dir / "top_candidates_e2e.csv", index=False)

        logger.info(f"\n  Top {top_n} candidates:")
        for i, c in enumerate(final_candidates[:10]):
            logger.info(
                f"    #{i+1}: {c['smiles'][:60]}... "
                f"T1-S1={c.get('T1_S1_gnn', '?'):.4f} "
                f"T2-S1={c.get('T2_S1_gnn', '?'):.4f} "
                f"Tanimoto={c.get('max_tanimoto', 0):.3f}"
            )

    # Export all novel SMILES
    if novel_molecules:
        all_novel = pd.DataFrame([
            {"smiles": m.get("canonical_smiles", m.get("smiles")),
             "max_tanimoto": m.get("max_tanimoto", 0),
             "scaffold": m.get("scaffold")}
            for m in novel_molecules
        ])
        all_novel.to_csv(output_dir / "all_novel_molecules.csv", index=False)

    # Summary
    summary = {
        "timestamp": timestamp,
        "n_training": int(len(data["X_train"])),
        "n_features": int(input_dim),
        "transfer_learning_used": qm9_file is not None,
        "gnn_trained": gnn is not None,
        "selfies_vae_trained": selfies_vae is not None,
        "n_generated": generation_stats.get("n_generated", 0),
        "n_valid": generation_stats.get("n_valid_smiles", 0),
        "n_novel": len(novel_molecules) if 'novel_molecules' in dir() else 0,
        "n_final": len(final_candidates),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  All results saved to {output_dir}/")
    logger.info("  Pipeline complete.")

    results["final_candidates"] = final_candidates
    results["summary"] = summary
    return results


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("e2e_pipeline.log")],
    )

    parser = argparse.ArgumentParser(description="End-to-End MR-TADF Inverse Design")
    parser.add_argument("--descriptor_file", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--qm9_file", type=str, default=None,
                        help="Optional: QM9 descriptor CSV for transfer learning")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_generate", type=int, default=1000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_e2e_pipeline(
        descriptor_file=args.descriptor_file,
        target_file=args.target_file,
        qm9_file=args.qm9_file,
        device=args.device,
        config={"n_generate": args.n_generate},
    )
