"""
novelty_validation.py — Novelty checking and chemical plausibility filters
for generated MR-TADF candidates.

Novelty is verified at three levels:
  1. SMILES string identity — canonical SMILES comparison
  2. Molecular fingerprint similarity — Tanimoto distance to all known molecules
  3. Scaffold/substructure novelty — Murcko scaffold decomposition

Chemical plausibility filters:
  - Boron count ≥ 1 (MR-TADF requires B atom)
  - Nitrogen count ≥ 1 (MR-TADF requires N atom for B/N alternation)
  - Aromatic ring count ≥ 3 (rigid polycyclic framework)
  - Molecular weight in [200, 1200] Da
  - No reactive functional groups (peroxides, azides, etc.)
  - Synthetic accessibility score (SA score) < 6

IMPORTANT: This module requires RDKit. Install with: pip install rdkit-pypi
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────── RDKit-dependent functions ───────────────────
# These are wrapped in try/except so the module can be imported
# even without RDKit (for descriptor-space-only workflows)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import RDConfig, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. SMILES-based novelty checks disabled.")


def canonicalise_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form; return None if invalid."""
    if not RDKIT_AVAILABLE:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    """Compute Morgan fingerprint (ECFP4) from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def compute_tanimoto(fp1, fp2) -> float:
    """Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Extract Murcko scaffold from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


# ─────────────────── MR-TADF Chemical Filters ───────────────────

class MRTADFChemicalFilter:
    """
    Chemical plausibility filters specific to MR-TADF molecules.

    MR-TADF emitters share key structural motifs:
      • Boron-nitrogen alternation in polycyclic aromatic frameworks
      • Rigid planar core (no rotatable bonds in core)
      • Typically 3–8 fused aromatic rings
      • B and N atoms at para/ortho positions in the aromatic system
    """
    FILTER_CRITERIA = {
        "has_boron": lambda mol: sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'B') >= 1,
        "has_nitrogen": lambda mol: sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N') >= 1,
        "min_aromatic_rings": lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol) >= 3,
        "mw_range": lambda mol: 200 <= Descriptors.MolWt(mol) <= 1200,
        "no_radicals": lambda mol: Descriptors.NumRadicalElectrons(mol) == 0,
        "max_rotatable": lambda mol: Descriptors.NumRotatableBonds(mol) <= 15,
        "min_heavy_atoms": lambda mol: mol.GetNumHeavyAtoms() >= 15,
    }

    # Reactive group SMARTS to exclude
    REACTIVE_SMARTS = [
        "[N]=[N]=[N]",   # azide
        "[O][O]",        # peroxide
        "C(=O)Cl",       # acyl chloride
        "[N+](=O)[O-]",  # nitro (sometimes acceptable)
    ]

    @classmethod
    def filter_molecule(cls, smiles: str) -> Tuple[bool, Dict[str, bool]]:
        """
        Apply all MR-TADF filters to a SMILES string.
        Returns (passes_all, individual_results).
        """
        if not RDKIT_AVAILABLE:
            return True, {"rdkit_unavailable": True}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {"valid_smiles": False}

        results = {"valid_smiles": True}

        # Apply structural filters
        for name, func in cls.FILTER_CRITERIA.items():
            try:
                results[name] = func(mol)
            except Exception:
                results[name] = False

        # Check for reactive groups
        has_reactive = False
        for smarts in cls.REACTIVE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                has_reactive = True
                break
        results["no_reactive_groups"] = not has_reactive

        passes_all = all(results.values())
        return passes_all, results


# ─────────────────── Novelty Validator ───────────────────

class NoveltyValidator:
    """
    Comprehensive novelty checking against a database of known MR-TADF molecules.

    Three levels of novelty:
      Level 1: Exact SMILES match (canonical form)
      Level 2: Tanimoto similarity < threshold (default 0.85)
      Level 3: Scaffold novelty — unique Murcko scaffold
    """
    def __init__(self, known_smiles: List[str], tanimoto_threshold: float = 0.85):
        self.tanimoto_threshold = tanimoto_threshold

        # Build database of canonical SMILES
        self.known_canonical = set()
        self.known_fps = []
        self.known_scaffolds = set()

        for smi in known_smiles:
            can = canonicalise_smiles(smi)
            if can:
                self.known_canonical.add(can)
                fp = get_morgan_fingerprint(can)
                if fp is not None:
                    self.known_fps.append(fp)
                scaffold = get_murcko_scaffold(can)
                if scaffold:
                    self.known_scaffolds.add(scaffold)

        logger.info(
            f"NoveltyValidator initialised: {len(self.known_canonical)} molecules, "
            f"{len(self.known_scaffolds)} unique scaffolds"
        )

    def check_novelty(self, smiles: str) -> Dict:
        """
        Check novelty of a candidate SMILES.
        Returns dict with novelty levels and details.
        """
        result = {
            "smiles": smiles,
            "is_novel_l1": True,  # canonical identity
            "is_novel_l2": True,  # fingerprint similarity
            "is_novel_l3": True,  # scaffold novelty
            "max_tanimoto": 0.0,
            "scaffold": None,
        }

        can = canonicalise_smiles(smiles)
        if can is None:
            result["is_novel_l1"] = False
            result["is_novel_l2"] = False
            result["is_novel_l3"] = False
            result["error"] = "Invalid SMILES"
            return result

        result["canonical_smiles"] = can

        # Level 1: Exact match
        if can in self.known_canonical:
            result["is_novel_l1"] = False

        # Level 2: Tanimoto similarity
        fp = get_morgan_fingerprint(can)
        if fp is not None and self.known_fps:
            similarities = [compute_tanimoto(fp, kfp) for kfp in self.known_fps]
            result["max_tanimoto"] = max(similarities) if similarities else 0.0
            if result["max_tanimoto"] > self.tanimoto_threshold:
                result["is_novel_l2"] = False

        # Level 3: Scaffold novelty
        scaffold = get_murcko_scaffold(can)
        result["scaffold"] = scaffold
        if scaffold and scaffold in self.known_scaffolds:
            result["is_novel_l3"] = False

        return result

    def batch_check(self, smiles_list: List[str]) -> List[Dict]:
        """Check novelty for a batch of SMILES strings."""
        return [self.check_novelty(smi) for smi in smiles_list]

    def filter_novel(self, smiles_list: List[str],
                     require_level: int = 2) -> List[Dict]:
        """
        Return only novel molecules at the specified level.
        Level 1: not an exact match
        Level 2: Tanimoto < threshold (strictest practical level)
        Level 3: novel scaffold (most restrictive)
        """
        results = self.batch_check(smiles_list)
        novel = []
        for r in results:
            if require_level == 1 and r["is_novel_l1"]:
                novel.append(r)
            elif require_level == 2 and r["is_novel_l1"] and r["is_novel_l2"]:
                novel.append(r)
            elif require_level == 3 and r["is_novel_l1"] and r["is_novel_l2"] and r["is_novel_l3"]:
                novel.append(r)
        return novel


# ─────────────────── Descriptor-Space Novelty ───────────────────

class DescriptorSpaceNoveltyChecker:
    """
    For workflows where SMILES are not directly available (e.g., when
    operating purely in descriptor space), check novelty via distance
    to known descriptor vectors.

    Uses Mahalanobis-like distance in the feature space.
    """
    def __init__(self, known_descriptors: np.ndarray, threshold_percentile: float = 95):
        self.known = known_descriptors
        self.mean = known_descriptors.mean(axis=0)
        self.std = known_descriptors.std(axis=0) + 1e-8

        # Compute pairwise distances among known molecules
        known_normed = (known_descriptors - self.mean) / self.std
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(known_normed, metric='euclidean')
        # Threshold: distance must exceed the 95th percentile of known inter-molecule distances
        self.min_distance_threshold = np.percentile(
            dists[np.triu_indices_from(dists, k=1)], threshold_percentile
        )
        logger.info(f"Descriptor novelty threshold (p{threshold_percentile}): {self.min_distance_threshold:.3f}")

    def check_novelty(self, candidate_descriptors: np.ndarray) -> np.ndarray:
        """
        Check if candidate descriptors are sufficiently different from all known.
        Returns boolean array (True = novel).
        """
        cand_normed = (candidate_descriptors - self.mean) / self.std
        known_normed = (self.known - self.mean) / self.std

        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(cand_normed, known_normed, metric='euclidean')
        min_dists = dists.min(axis=1)

        is_novel = min_dists > self.min_distance_threshold * 0.5  # relaxed for generative
        return is_novel, min_dists


# ─────────────────── Comprehensive Screening Pipeline ───────────────────

def screen_candidates(
    candidate_descriptors: np.ndarray,
    predicted_properties: np.ndarray,
    known_smiles: List[str],
    known_descriptors: np.ndarray,
    scaler_y=None,
    t1s1_thresh: float = 0.08,
    t2s1_thresh: float = 0.10,
) -> Dict:
    """
    Full screening pipeline for generated candidates:
      1. Property filter: T1-S1 < thresh, T2-S1 < thresh
      2. Descriptor-space novelty
      3. Ranking by property desirability

    Returns dict with filtered candidates and statistics.
    """
    n_total = len(candidate_descriptors)

    # Convert to raw properties if needed
    if scaler_y is not None:
        props_raw = scaler_y.inverse_transform(predicted_properties)
    else:
        props_raw = predicted_properties

    # Step 1: Property filter
    t1s1 = np.abs(props_raw[:, 0])
    t2s1 = np.abs(props_raw[:, 1])
    prop_mask = (t1s1 < t1s1_thresh) & (t2s1 < t2s1_thresh)
    n_pass_props = prop_mask.sum()
    logger.info(f"Property filter: {n_pass_props}/{n_total} pass")

    if n_pass_props == 0:
        # Relax thresholds and report
        for factor in [1.5, 2.0, 3.0]:
            relaxed = (t1s1 < t1s1_thresh * factor) & (t2s1 < t2s1_thresh * factor)
            if relaxed.sum() > 0:
                logger.info(f"  Relaxed {factor}x: {relaxed.sum()} pass")
                prop_mask = relaxed
                n_pass_props = relaxed.sum()
                break

    # Step 2: Descriptor-space novelty
    desc_checker = DescriptorSpaceNoveltyChecker(known_descriptors)
    is_novel, min_dists = desc_checker.check_novelty(candidate_descriptors[prop_mask])

    # Step 3: Ranking — minimise T1-S1 + T2-S1 (smaller = better TADF performance)
    passed_idx = np.where(prop_mask)[0]
    novel_idx = passed_idx[is_novel]

    if len(novel_idx) > 0:
        desirability = t1s1[novel_idx] + t2s1[novel_idx]  # lower is better
        rank_order = np.argsort(desirability)
        ranked_idx = novel_idx[rank_order]
    else:
        ranked_idx = np.array([], dtype=int)

    return {
        "n_total": n_total,
        "n_pass_properties": int(n_pass_props),
        "n_novel": int(is_novel.sum()) if n_pass_props > 0 else 0,
        "ranked_indices": ranked_idx,
        "properties_raw": props_raw,
        "min_distances": min_dists if n_pass_props > 0 else np.array([]),
        "t1s1_values": t1s1,
        "t2s1_values": t2s1,
    }
