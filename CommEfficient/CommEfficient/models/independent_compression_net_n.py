"""
Independent Compression for any number of modalities — summation fusion
+ original FetchSGD.

Same idea as ``independent_compression_net_4.py``, with the number of branches
taken from the length of ``mod_dims``: one ``FeaExtractor``/``FeaRefiner`` pair
per modality, refined features fused by plain element-wise **sum**, then an
integrator MLP and a linear classifier.

Pipeline
--------
  1. FeaExtractor + FeaRefiner per modality  →  f'_0 … f'_{N-1}
  2. Summation fusion                        →  fused = sum(f'_i)
  3. Integrator MLP                          →  H
  4. Classifier                              →  logits
"""

import torch.nn as nn

from .multimodal_net import FeaExtractor, FeaRefiner

__all__ = ["IndependentCompressionN"]


class IndependentCompressionN(nn.Module):
    """N-branch multimodal net with **summation** fusion."""

    def __init__(
        self,
        mod_dims,
        feat_dim=512,
        num_classes=10,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.mod_dims = tuple(int(d) for d in mod_dims)
        self.extractors = nn.ModuleList(
            [FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims]
        )
        self.refiners = nn.ModuleList(
            [FeaRefiner(feat_dim) for _ in self.mod_dims]
        )

        self.integrator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *mods):
        fused = 0
        for extractor, refiner, x in zip(self.extractors, self.refiners, mods):
            f = extractor(x)
            fused = fused + f * refiner(f)
        H = self.integrator(fused)
        return self.classifier(H), H
