# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Sphere_VAE model with continuous spherical latent variables."""

from contextlib import nullcontext
from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F

from modules import PowerSphericalDistribution, l2_norm
from modules import SEANetEncoder, SEANetDecoder, transformer
from utils.compile import no_compile


@dataclass
class Sphere_VAE_result:
    vae_latent: torch.Tensor
    recon: torch.Tensor
    metrics: dict = field(default_factory=dict)


class Sphere_VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        latent_dimension: int = 64,
        d_model: int = 512, 
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.latent_dimension = latent_dimension
        self.feature_dim = d_model
        self.quant_proj = nn.Linear(self.feature_dim, self.latent_dimension+1)
        self.post_quant_proj = nn.Linear(self.latent_dimension, self.feature_dim)
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder
        
    def normalize_latent(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:


        x = l2_norm(x, dim=dim)
        x = x * (self.latent_dimension**0.5)
        return x
    
    @property
    def _context_for_encoder_decoder(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()
        
    def encode_features(self, x):
        assert x.dim() == 3
        length = x.shape[-1]
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)

        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)

        return emb
    
    def encode_mu(self, x):
        emb = self.encode_features(x)
        q = self.quant_proj(emb.transpose(1, 2))
        mu = q[..., :-1]
        mu = l2_norm(mu)
        mu = mu * (self.latent_dimension ** 0.5)
        return mu.transpose(1, 2)
    

    def encode(self, x):
        assert x.dim() == 3
        length = x.shape[-1]
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        x = self.quant_proj(emb.transpose(1, 2))
        mu = x[..., :-1]
        kappa = x[..., -1]
        mu = l2_norm(mu)
        kappa = F.softplus(kappa) + 1.0
        qz = PowerSphericalDistribution(mu, kappa)
        loss = qz.kl_to_uniform()
        x = qz.rsample()
        x = x * (self.latent_dimension**0.5)
        x = x.transpose(1, 2)
        return x, loss.mean()

    def decode(self, x):
        assert x.dim() == 3
        length = x.shape[-1]
        
        x = self.post_quant_proj(x.transpose(1, 2)).transpose(1, 2)
        if self.decoder_transformer is not None:
            (x,) = self.decoder_transformer(x)
            
        with self._context_for_encoder_decoder:
            dec = self.decoder(x)
        return dec
    
    def forward(self, x: torch.Tensor) -> Sphere_VAE_result:

        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}        
        vae_latent, kl_loss = self.encode(x)
        extra_metrics["kl_loss"] = kl_loss
        recon = self.decode(vae_latent)
        
        return Sphere_VAE_result(vae_latent, recon, extra_metrics)


def build_model(
    d_model: int,
    seanet_kwargs,
    transformer_kwargs,
    latent_dimension: int = 64,
) -> Sphere_VAE:
    """Build Sphere_VAE."""
    return Sphere_VAE(
        encoder=SEANetEncoder(**seanet_kwargs),
        decoder=SEANetDecoder(**seanet_kwargs),
        encoder_transformer=transformer.ProjectedTransformer(**transformer_kwargs),
        decoder_transformer=transformer.ProjectedTransformer(**transformer_kwargs),
        d_model=d_model,
        latent_dimension=latent_dimension,
    )
