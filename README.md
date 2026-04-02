# Inverted Design Pipeline for Novel MRTADF design

<img width="734" height="875" alt="image" src="https://github.com/user-attachments/assets/c9f59282-3888-4089-9a2b-bffc72649f7a" />

## e2e_pipeline.py :End-to-end MR-TADF molecular generation pipeline

Integrates all four components into a single workflow:

  1. QM9 pre-training + domain adaptation  (transfer_learning.py)
  2. GNN property predictor               (gnn_predictor.py)
  3. SELFIES conditional VAE generation    (selfies_generator.py)
  4. Novelty & chemical-plausibility gate  (novelty_validation.py)


Workflow:
```text
  ┌──────────────────────────────────────────────────┐
  │ Phase A: Pre-train backbone on QM9 descriptors   │
  │ Phase B: MMD domain adaptation → MR-TADF         │
  │ Phase C: Progressive fine-tune on MR-TADF        │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Train GNN predictor on MR-TADF molecular graphs  │
  │ (operates on SMILES → graph, independent of      │
  │  descriptor pipeline; provides ensemble scoring) │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Train SELFIES cVAE on MR-TADF SMILES + targets   │
  │ → generates novel valid molecules directly       │
  └────────────────────┬─────────────────────────────┘
                       │
  ┌────────────────────┴─────────────────────────────┐
  │ Conditional generation:                          │
  │   Set target y* = [T1-S1=0.04, T2-S1=0.05, ...]  │
  │   Sample z ~ N(0,I), decode to SELFIES → SMILES  │
  │   Score with GNN + transfer-learned predictor    │
  │   Filter: property compliance + novelty + chem.  │
  │   Rank and export top candidates                 │
  └──────────────────────────────────────────────────┘
```

## transfer_learning.py :QM9-pretrained backbone with domain-adaptive fine-tuning


### Scientific rationale:

With only 176 MR-TADF molecules, a randomly-initialised deep network cannot   learn generalisable structure–property relationships.  QM9 provides ~134 k   small organic molecules with DFT-computed HOMO, LUMO, gap, dipole, and atomisation energies.  Although QM9 molecules are smaller (≤9 heavy atoms) than typical MR-TADF emitters (40–130 heavy atoms), the *electronic-structure descriptors* (Moreau-Broto autocorrelations, ETA indices, PEOE_VSA, etc.) share the same feature space, so a backbone pre-trained on QM9 learns reusable descriptor→property mappings that transfer.

### Strategy (3 phases):

Phase 1 — Pre-train on QM9
  
• Compute the same 2870 Mordred/RDKit descriptors for all QM9 molecules

• Train backbone to predict HOMO, LUMO, gap, μ, α (5 tasks)

• This teaches the model generic "descriptor → electronic property" patterns

Phase 2 — Domain-adaptive alignment
  
• Use Maximum Mean Discrepancy (MMD) loss to align the latent representations of QM9 and MR-TADF descriptor distributions

• This closes the domain gap arising from molecular-size differences

Phase 3 — Task-specific fine-tuning on MR-TADF
  
• Replace the 5-target QM9 head with the 6-target MR-TADF head

• Apply progressive unfreezing: first train only the new head, then unfreeze top blocks, then all parameters

• Use discriminative learning rates (lower LR for early layers)


## selfies_generator.py :SELFIES-based molecular generation for MR-TADF

### Scientific rationale:

SMILES strings can represent invalid molecules (unmatched brackets, impossible valences).  SELFIES (Self-Referencing Embedded Strings, Krenn et al., Mach. Learn.: Sci. Technol. 2020) guarantee 100%   syntactic validity: every SELFIES string decodes to a valid molecular graph.  This eliminates the need for post-hoc validity filtering and increases the effective yield of generative models by 3–10×.

Architecture — Conditional SELFIES VAE:

Encoder:  SELFIES tokens → Transformer → latent z
Decoder:  z + target properties → Transformer → SELFIES tokens

The model learns p(molecule | properties), enabling:
1. Conditional generation: specify desired T1-S1, T2-S1 → sample z → decode SELFIES
2. Interpolation: blend two molecules in latent space
3. Optimisation: gradient ascent on z to optimise predicted properties

Teacher forcing during training; autoregressive sampling at generation.

Key design choices for MR-TADF:

• Vocabulary includes [B], [N], [=N], [#N], [O], [S], [F] explicitly to ensure the model can represent B/N frameworks

• Maximum sequence length 256 tokens (covers the largest MR-TADF SMILES)

• Positional encoding enables attention over the full sequence

• Property conditioning is injected at every decoder layer (not just init)

## gnn_predictor.py: Graph Neural Network property predictor for MR-TADF

### Scientific rationale:

Molecular descriptors are hand-crafted projections that discard 3D and topological information.  A GNN operates directly on the molecular graph (atoms = nodes, bonds = edges), preserving the full connectivity that
determines electronic structure.  For MR-TADF emitters, the *alternating B/N substitution pattern* within the aromatic framework is the defining structural motif — a GNN naturally encodes this through message passing
along the covalent bond network.

Architecture:

Atom featuriser → N × MPNN layers → global readout → task heads

• Atom features: atomic number, formal charge, hybridisation, aromaticity, number of Hs, degree, is_in_ring, atomic mass (one-hot + continuous)

• Bond features: bond type, conjugation, ring membership, stereochem

• MPNN uses edge-conditioned message passing (Gilmer et al., ICML 2017) with GRU-based node update for stable deep propagation

• Global readout: Set2Set attention pooling (Vinyals et al., NeurIPS 2016)

• Multi-task heads identical to the descriptor-based predictor

The GNN can replace the descriptor-based predictor entirely, or serve as an ensemble member alongside it.

### Required Library 

conda config --env --set channel_priority strict

conda install -y -c conda-forge \
  "numpy=1.26.*" pandas scikit-learn openpyxl selfies rdkit mordred

conda install -y pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch_geometric

### How to run

```text
python e2e_pipeline.py \
  --descriptor_file modified_skewness_filtered_data_updated_smile.xlsx \
  --target_file target5.xlsx \
  --qm9_file qm9_descriptors.csv \
  --device cuda \
  --n_generate 1000
```
