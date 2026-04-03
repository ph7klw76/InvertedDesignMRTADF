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
  "numpy=1.26.*" pandas scikit-learn openpyxl selfies rdkit mordred "sympy=1.13.1"

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

# Generating Descriptor for QM9

## Problem
mordred.Calculator.pandas() on all molecules at once and padelpy.padeldescriptor() on a single giant SDF file. This fails because: PaDEL's Java JVM runs out of heap memory on large batches and silently freezes; Mordred infinite-loops on certain polycyclic topologies with no timeout mechanism; padelpy blocks the entire Python process so a single hung molecule kills everything; and writing one CSV at the end means any crash — even after 8 hours of work — loses all computed data.

## Solution
The CheckpointDB class uses SQLite in WAL (Write-Ahead Logging) journal mode, which guarantees that a crash mid-write cannot corrupt the database. Each molecule gets one row with three independent status flags (mordred_done, rdkit_done, padel_done) and descriptor values stored as zlib-compressed JSON blobs. After computing descriptors for each molecule, the result is committed atomically , so even a power failure preserves every molecule completed before that instant. On restart, the pipeline queries WHERE mordred_done = 0 to find only unprocessed molecules and skips everything already computed.
The TimeoutExecutor wraps every computation in a ProcessPoolExecutor(max_workers=1) with a hard timeout. When Mordred hangs on a ring topology or PaDEL's Java process freezes, the subprocess is terminated after the deadline (120s for Mordred, 45s for PaDEL). This is the only reliable way to kill a hung Java JVM from Python. The molecule is marked as failed (done = -1) with an error message, and the pipeline moves to the next one.
PaDEL is processed in micro-batches of 25 molecules (not 134,000 in one SDF). If a batch fails, each molecule in that batch is retried individually via padelpy.from_smiles() with its own timeout. This two-level fallback means a single toxic molecule cannot bring down the entire batch. (robust_descriptor_pipeline.py)

<img width="610" height="660" alt="image" src="https://github.com/user-attachments/assets/4efc28e3-4d49-4536-9809-65b429022e23" />

## How to use
Java is also required for PaDEL , download from https://adoptium.net/ (Temurin JDK 17 or 21), check "Add to PATH" during install, then verify with java -version in a fresh Anaconda Prompt.
How to use in Spyder:

Open robust_descriptor_pipeline.py, scroll to the bottom, edit the four paths, set max_molecules=100 to test, and press F5. If it works, set max_molecules=None for all QM9 molecules and press F5 again. If it crashes at any point : power failure, Java OOM, Mordred hang, accidental window close , just press F5 again. The pipeline reopens the checkpoint database, reports how many molecules were completed in each stage, and resumes from exactly where it stopped.

## Resource estimates for full QM9 (134k molecules):
The estimated total time is roughly 37 hours with 4 PaDEL threads, the SQLite checkpoint database will be approximately 314 MB, and the final CSV will be around 3 GB. You can monitor progress via the live progress bar with throughput and ETA, and a descriptor_pipeline.log file captures every failure with full details. The pipeline also runs gc.collect() every 200 molecules to prevent memory leaks during long runs.
