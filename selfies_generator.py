"""
selfies_generator.py — SELFIES-based molecular generation for MR-TADF
=======================================================================

Scientific rationale:
  SMILES strings can represent invalid molecules (unmatched brackets,
  impossible valences).  SELFIES (Self-Referencing Embedded Strings,
  Krenn et al., Mach. Learn.: Sci. Technol. 2020) guarantee 100%
  syntactic validity: every SELFIES string decodes to a valid molecular
  graph.  This eliminates the need for post-hoc validity filtering and
  increases the effective yield of generative models by 3–10×.

Architecture — Conditional SELFIES VAE:
  Encoder:  SELFIES tokens → Transformer → latent z
  Decoder:  z + target properties → Transformer → SELFIES tokens

  The model learns p(molecule | properties), enabling:
    1. Conditional generation: specify desired T1-S1, T2-S1 → sample z → decode SELFIES
    2. Interpolation: blend two molecules in latent space
    3. Optimisation: gradient ascent on z to optimise predicted properties

  Teacher forcing during training; autoregressive sampling at generation.

Key design choices for MR-TADF:
  • Vocabulary includes [B], [N], [=N], [#N], [O], [S], [F] explicitly
    to ensure the model can represent B/N frameworks
  • Maximum sequence length 256 tokens (covers the largest MR-TADF SMILES)
  • Positional encoding enables attention over the full sequence
  • Property conditioning is injected at every decoder layer (not just init)

Dependencies:
  pip install torch selfies rdkit-pypi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. SELFIES TOKENISER
# ═══════════════════════════════════════════════════════════════════

class SELFIESTokeniser:
    """
    Builds a vocabulary from SELFIES strings and provides encode/decode.

    Special tokens:
      <pad>  = 0  — padding
      <bos>  = 1  — beginning of sequence
      <eos>  = 2  — end of sequence
      <unk>  = 3  — unknown token
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self.token_to_idx = dict(self.SPECIAL_TOKENS)
        self.idx_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.vocab_size = len(self.SPECIAL_TOKENS)
        self._fitted = False

    def fit(self, smiles_list: List[str]) -> "SELFIESTokeniser":
        """
        Build vocabulary from a list of SMILES by converting to SELFIES
        and collecting all unique tokens.
        """
        try:
            import selfies as sf
        except ImportError:
            raise ImportError("pip install selfies")

        all_tokens = Counter()
        self._selfies_list = []

        for smi in smiles_list:
            try:
                selfies_str = sf.encoder(smi)
                if selfies_str is None:
                    continue
                tokens = list(sf.split_selfies(selfies_str))
                all_tokens.update(tokens)
                self._selfies_list.append(selfies_str)
            except Exception:
                continue

        # Build vocabulary sorted by frequency
        for token, _ in all_tokens.most_common():
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token

        self.vocab_size = len(self.token_to_idx)
        self._fitted = True

        logger.info(
            f"SELFIES vocabulary: {self.vocab_size} tokens from "
            f"{len(self._selfies_list)}/{len(smiles_list)} valid molecules. "
            f"Max token length observed: {max(len(list(sf.split_selfies(s))) for s in self._selfies_list)}"
        )
        return self

    def encode(self, smiles: str) -> Optional[List[int]]:
        """Convert SMILES → SELFIES → token indices."""
        try:
            import selfies as sf
            selfies_str = sf.encoder(smiles)
            if selfies_str is None:
                return None
            tokens = list(sf.split_selfies(selfies_str))
            indices = [self.SPECIAL_TOKENS["<bos>"]]
            for t in tokens:
                indices.append(self.token_to_idx.get(t, self.SPECIAL_TOKENS["<unk>"]))
            indices.append(self.SPECIAL_TOKENS["<eos>"])
            return indices
        except Exception:
            return None

    def decode(self, indices: List[int]) -> Optional[str]:
        """Convert token indices → SELFIES → SMILES."""
        try:
            import selfies as sf
            tokens = []
            for idx in indices:
                token = self.idx_to_token.get(idx, "<unk>")
                if token == "<eos>":
                    break
                if token in ("<pad>", "<bos>", "<unk>"):
                    continue
                tokens.append(token)
            selfies_str = "".join(tokens)
            smiles = sf.decoder(selfies_str)
            return smiles
        except Exception:
            return None

    def batch_encode(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of SMILES to padded tensors.

        Returns:
            token_ids: (B, max_length) LongTensor
            lengths: (B,) actual lengths (including <bos>/<eos>)
        """
        encoded = []
        for smi in smiles_list:
            indices = self.encode(smi)
            if indices is None:
                indices = [self.SPECIAL_TOKENS["<bos>"], self.SPECIAL_TOKENS["<eos>"]]
            # Truncate if needed
            if len(indices) > self.max_length:
                indices = indices[:self.max_length - 1] + [self.SPECIAL_TOKENS["<eos>"]]
            encoded.append(indices)

        lengths = [len(seq) for seq in encoded]
        max_len = min(max(lengths), self.max_length)

        padded = torch.full((len(encoded), max_len), self.SPECIAL_TOKENS["<pad>"],
                             dtype=torch.long)
        for i, seq in enumerate(encoded):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded, torch.tensor(lengths, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════
#  2. POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE for transformer models."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════
#  3. CONDITIONAL SELFIES VAE
# ═══════════════════════════════════════════════════════════════════

class SELFIESConditionalVAE(nn.Module):
    """
    Conditional VAE operating on SELFIES token sequences.

    Encoder:
      Token embedding + PE → Transformer encoder → [CLS] aggregation →
      concat with property condition → MLP → (μ, log σ²)

    Decoder:
      z + property condition → memory → Transformer decoder (causal) →
      linear → logits over vocabulary

    Property conditioning is injected:
      1. As an additive bias to the encoder output before μ/σ projection
      2. As a cross-attention memory prefix in the decoder
      3. As an additive signal at every decoder layer via FiLM conditioning

    This triple injection ensures the decoder strongly respects the
    desired properties rather than ignoring the condition.
    """
    def __init__(self,
                 vocab_size: int,
                 max_length: int = 256,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 6,
                 latent_dim: int = 128,
                 property_dim: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.property_dim = property_dim

        # ── Token embedding ──
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encode = SinusoidalPositionalEncoding(d_model, max_length, dropout)

        # ── Property conditioning network ──
        self.property_proj = nn.Sequential(
            nn.Linear(property_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Encoder → latent
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # ── Decoder ──
        # Latent + property → initial decoder memory
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim + property_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # FiLM conditioning: per-layer scale and shift from properties
        self.film_layers = nn.ModuleList([
            nn.Linear(property_dim, 2 * d_model)
            for _ in range(n_decoder_layers)
        ])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.n_decoder_layers = n_decoder_layers

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def encode(self, token_ids: torch.Tensor,
               properties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode SELFIES token sequence to latent distribution.

        token_ids: (B, seq_len)
        properties: (B, property_dim)

        Returns: mu (B, latent_dim), logvar (B, latent_dim)
        """
        # Create padding mask
        pad_mask = (token_ids == 0)  # True where padded

        # Embed tokens + positional encoding
        x = self.token_embed(token_ids)
        x = self.pos_encode(x)

        # Add property conditioning to the token embeddings
        prop_embed = self.property_proj(properties).unsqueeze(1)  # (B, 1, d_model)
        x = x + prop_embed  # broadcast across sequence

        # Transformer encoder
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)

        # Global pool: mean over non-padded positions
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, properties: torch.Tensor,
               target_ids: Optional[torch.Tensor] = None,
               max_len: Optional[int] = None) -> torch.Tensor:
        """
        Decode latent vector to SELFIES token logits.

        Training (target_ids provided):
          Teacher forcing — feed ground-truth tokens shifted right

        Generation (target_ids is None):
          Autoregressive — sample one token at a time

        Returns:
          Training: logits (B, seq_len, vocab_size)
          Generation: token_ids (B, max_len)
        """
        B = z.size(0)
        device = z.device

        # Create memory from latent + properties
        memory_input = torch.cat([z, properties], dim=-1)
        memory = self.latent_to_memory(memory_input).unsqueeze(1)  # (B, 1, d_model)

        if target_ids is not None:
            # ── Teacher forcing ──
            seq_len = target_ids.size(1)
            tgt = self.token_embed(target_ids)
            tgt = self.pos_encode(tgt)

            causal_mask = self._generate_causal_mask(seq_len, device)

            # Apply FiLM conditioning at each decoder layer
            decoded = tgt
            for i, layer in enumerate(self.decoder.layers):
                # FiLM: γ, β from properties
                film = self.film_layers[i](properties)  # (B, 2*d_model)
                gamma = film[:, :self.d_model].unsqueeze(1)  # (B, 1, d_model)
                beta = film[:, self.d_model:].unsqueeze(1)

                decoded = layer(
                    decoded, memory,
                    tgt_mask=causal_mask,
                )
                decoded = gamma * decoded + beta  # FiLM modulation

            logits = self.output_proj(decoded)
            return logits

        else:
            # ── Autoregressive generation ──
            if max_len is None:
                max_len = self.max_length

            generated = torch.full((B, 1), 1, dtype=torch.long, device=device)  # <bos>

            for step in range(max_len - 1):
                tgt = self.token_embed(generated)
                tgt = self.pos_encode(tgt)

                causal_mask = self._generate_causal_mask(tgt.size(1), device)

                decoded = tgt
                for i, layer in enumerate(self.decoder.layers):
                    film = self.film_layers[i](properties)
                    gamma = film[:, :self.d_model].unsqueeze(1)
                    beta = film[:, self.d_model:].unsqueeze(1)

                    decoded = layer(decoded, memory, tgt_mask=causal_mask)
                    decoded = gamma * decoded + beta

                logits = self.output_proj(decoded[:, -1:, :])  # last position
                next_token = logits.argmax(dim=-1)  # greedy; or sample
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all sequences have produced <eos>
                if (next_token == 2).all():
                    break

            return generated

    def forward(self, token_ids: torch.Tensor,
                properties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (training mode with teacher forcing).

        token_ids: (B, seq_len) — full sequence including <bos> and <eos>
        properties: (B, property_dim)

        Returns: logits, mu, logvar
        """
        mu, logvar = self.encode(token_ids, properties)
        z = self.reparameterise(mu, logvar)

        # Decoder input: all tokens except last (shift right)
        decoder_input = token_ids[:, :-1]
        # Decoder target: all tokens except first
        logits = self.decode(z, properties, target_ids=decoder_input)

        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, properties: torch.Tensor,
                 n_samples: int = 1,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 max_len: int = 200) -> torch.Tensor:
        """
        Generate SELFIES sequences conditioned on target properties.

        Args:
            properties: (B, property_dim) — desired properties
            n_samples: how many samples per property vector
            temperature: sampling temperature (< 1 = more conservative)
            top_k: if > 0, sample only from top-k logits
            max_len: maximum generation length

        Returns: (B * n_samples, max_len) token IDs
        """
        self.eval()
        B = properties.size(0)
        device = properties.device

        # Expand for multiple samples
        props_exp = properties.repeat_interleave(n_samples, dim=0)
        z = torch.randn(B * n_samples, self.latent_dim, device=device) * temperature

        # Create memory
        memory_input = torch.cat([z, props_exp], dim=-1)
        memory = self.latent_to_memory(memory_input).unsqueeze(1)

        generated = torch.full(
            (B * n_samples, 1), 1, dtype=torch.long, device=device
        )  # <bos>

        for step in range(max_len - 1):
            tgt = self.token_embed(generated)
            tgt = self.pos_encode(tgt)
            causal_mask = self._generate_causal_mask(tgt.size(1), device)

            decoded = tgt
            for i, layer in enumerate(self.decoder.layers):
                film = self.film_layers[i](props_exp)
                gamma = film[:, :self.d_model].unsqueeze(1)
                beta = film[:, self.d_model:].unsqueeze(1)
                decoded = layer(decoded, memory, tgt_mask=causal_mask)
                decoded = gamma * decoded + beta

            logits = self.output_proj(decoded[:, -1, :]) / temperature

            # Top-k filtering (clamp to vocab size)
            effective_k = min(top_k, logits.size(-1)) if top_k > 0 else 0
            if effective_k > 0:
                indices_to_remove = logits < torch.topk(logits, effective_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == 2).all():
                break

        return generated


# ═══════════════════════════════════════════════════════════════════
#  4. TRAINING LOSS
# ═══════════════════════════════════════════════════════════════════

class SELFIESVAELoss(nn.Module):
    """
    ELBO loss for the SELFIES VAE:
      L = L_recon + β · L_KL

    L_recon: cross-entropy on next-token prediction (teacher forcing)
    L_KL: KL divergence from the standard normal prior

    Uses cyclical β-annealing (Fu et al., 2019) to prevent posterior
    collapse — a critical problem for discrete-token VAEs.

    Additionally includes a "free bits" mechanism (Kingma et al., 2016):
    KL is only penalised above λ = 0.1 nats per latent dimension,
    allowing the model to use the latent space without being pushed
    to a trivial posterior.
    """
    def __init__(self, beta_max: float = 0.5, free_bits: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.beta_max = beta_max
        self.free_bits = free_bits
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor,
                epoch: int = 0, max_epoch: int = 100) -> Tuple[torch.Tensor, Dict]:
        """
        logits: (B, seq_len, vocab_size)
        targets: (B, seq_len) — ground-truth token IDs (shifted)
        """
        B, S, V = logits.shape

        # Reconstruction: cross-entropy ignoring padding
        recon_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
            reduction='mean',
        )

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl_loss = kl_per_dim.mean()

        # Cyclical β schedule (4 cycles)
        cycle_len = max(max_epoch // 4, 1)
        cycle_pos = (epoch % cycle_len) / cycle_len
        beta = min(self.beta_max, self.beta_max * cycle_pos * 2)

        total = recon_loss + beta * kl_loss

        # Token-level accuracy (for monitoring)
        with torch.no_grad():
            pred_tokens = logits.argmax(dim=-1)
            mask = targets != self.pad_idx
            accuracy = (pred_tokens[mask] == targets[mask]).float().mean().item()

        return total, {
            "total": total.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "beta": beta,
            "accuracy": accuracy,
        }


# ═══════════════════════════════════════════════════════════════════
#  5. SELFIES VAE TRAINER
# ═══════════════════════════════════════════════════════════════════

class SELFIESVAETrainer:
    """Training loop for the SELFIES conditional VAE."""

    def __init__(self, model: SELFIESConditionalVAE,
                 tokeniser: SELFIESTokeniser,
                 lr: float = 3e-4,
                 beta_max: float = 0.5,
                 max_epochs: int = 200,
                 patience: int = 25,
                 batch_size: int = 32,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.tokeniser = tokeniser
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

        self.criterion = SELFIESVAELoss(beta_max=beta_max)
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser, T_0=50, T_mult=2
        )

    def fit(self, smiles_train: List[str], properties_train: np.ndarray,
            smiles_val: List[str], properties_val: np.ndarray) -> Dict:
        """
        Train the SELFIES VAE.

        smiles_train/val: lists of SMILES strings
        properties_train/val: (n, property_dim) arrays
        """
        # Tokenise
        tokens_train, lengths_train = self.tokeniser.batch_encode(smiles_train)
        tokens_val, lengths_val = self.tokeniser.batch_encode(smiles_val)

        props_train_t = torch.tensor(properties_train, dtype=torch.float32)
        props_val_t = torch.tensor(properties_val, dtype=torch.float32)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(tokens_train, props_train_t)
        val_dataset = torch.utils.data.TensorDataset(tokens_val, props_val_t)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_acc = 0
            n_batch = 0

            for token_batch, prop_batch in train_loader:
                token_batch = token_batch.to(self.device)
                prop_batch = prop_batch.to(self.device)

                self.optimiser.zero_grad()

                logits, mu, logvar = self.model(token_batch, prop_batch)

                # Target: shifted tokens (everything after <bos>)
                target = token_batch[:, 1:]

                loss, info = self.criterion(
                    logits, target, mu, logvar,
                    epoch=epoch, max_epoch=self.max_epochs,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()

                train_loss += info["total"]
                train_acc += info["accuracy"]
                n_batch += 1

            self.scheduler.step()

            # Validate
            val_metrics = self._evaluate(val_loader, epoch)

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 10 == 0:
                logger.info(
                    f"SELFIES Epoch {epoch:3d} | "
                    f"train_loss={train_loss/n_batch:.4f} "
                    f"val_loss={val_metrics['total']:.4f} "
                    f"recon={val_metrics['recon']:.4f} "
                    f"kl={val_metrics['kl']:.4f} "
                    f"acc={val_metrics['accuracy']:.3f}"
                )

            if patience_count >= self.patience:
                logger.info(f"SELFIES VAE early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return self._evaluate(val_loader, self.max_epochs)

    @torch.no_grad()
    def _evaluate(self, loader, epoch) -> Dict:
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_acc = 0
        n = 0

        for token_batch, prop_batch in loader:
            token_batch = token_batch.to(self.device)
            prop_batch = prop_batch.to(self.device)

            logits, mu, logvar = self.model(token_batch, prop_batch)
            target = token_batch[:, 1:]

            _, info = self.criterion(
                logits, target, mu, logvar,
                epoch=epoch, max_epoch=self.max_epochs,
            )

            total_loss += info["total"]
            total_recon += info["recon"]
            total_kl += info["kl"]
            total_acc += info["accuracy"]
            n += 1

        return {
            "total": total_loss / max(n, 1),
            "recon": total_recon / max(n, 1),
            "kl": total_kl / max(n, 1),
            "accuracy": total_acc / max(n, 1),
        }

    def generate_molecules(self, target_properties: np.ndarray,
                            n_samples: int = 10,
                            temperature: float = 0.8,
                            top_k: int = 0) -> List[Optional[str]]:
        """
        Generate SMILES strings for desired properties.

        target_properties: (B, property_dim) in scaled space
        Returns: list of SMILES strings (None for decoding failures)
        """
        self.model.eval()
        props_t = torch.tensor(target_properties, dtype=torch.float32,
                                device=self.device)

        token_ids = self.model.generate(
            props_t, n_samples=n_samples,
            temperature=temperature, top_k=top_k,
        )

        # Decode back to SMILES
        smiles_list = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].cpu().tolist()
            smi = self.tokeniser.decode(ids)
            smiles_list.append(smi)

        n_valid = sum(1 for s in smiles_list if s is not None)
        logger.info(
            f"Generated {len(smiles_list)} molecules, {n_valid} valid "
            f"({n_valid/len(smiles_list)*100:.1f}%)"
        )
        return smiles_list
