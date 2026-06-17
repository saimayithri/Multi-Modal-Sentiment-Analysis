import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=5, layers=2,
                 relu_dropout=0.1, embed_dropout=0.3,
                 attn_dropout=0.25, res_dropout=0.1):
        super(Classifier, self).__init__()
        self.bone = TransformerEncoder(embed_dim=in_dim, num_heads=num_heads,
                                       layers=layers, attn_dropout=attn_dropout, res_dropout=res_dropout,
                                       relu_dropout=relu_dropout, embed_dropout=embed_dropout)
        self.proj1 = nn.Linear(in_dim, in_dim)
        self.out_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.bone(x)
        x = x[0]
        x = F.relu(self.proj1(x))
        x = self.out_layer(x)
        return x


class UnimodalAuxiliaryHeads(nn.Module):
    def __init__(self, output_dim, num_mod, proj_dim=30, **kwargs):
        super(UnimodalAuxiliaryHeads, self).__init__()
        self.num_mod = num_mod
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(proj_dim // 2, output_dim)
            )
            for _ in range(self.num_mod)
        ])

    def forward(self, x_cls_tokens):
        return [self.classifiers[i](x_cls_tokens[i]) for i in range(self.num_mod)]


class GateAttention(nn.Module):
    """Learnable attention-based gating mechanism for modality fusion.
    
    Unlike the previous detached gating, gradients flow through this module
    back to the modality encoders, allowing the model to learn how to 
    weight each modality adaptively per sample.
    """
    def __init__(self, proj_dim, num_mod, temperature=1.0):
        super(GateAttention, self).__init__()
        self.temperature = temperature
        # Shared query vector for attention-based gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.Tanh(),
            nn.Linear(proj_dim // 2, 1)
        )
        self.num_mod = num_mod

    def forward(self, h_cls_tokens):
        """
        Args:
            h_cls_tokens: list of [Batch, Dim] tensors, one per modality
        Returns:
            gates: [Batch, num_mod] softmax weights (gradients flow through)
        """
        stacked = torch.stack(h_cls_tokens, dim=1)  # [B, num_mod, D]
        gate_logits = self.gate_mlp(stacked).squeeze(-1)  # [B, num_mod]
        gates = F.softmax(gate_logits / self.temperature, dim=1)  # [B, num_mod]
        return gates


class MSAModel(nn.Module):
    def __init__(self, output_dim, orig_dim, proj_dim=40, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25, **kwargs):
        super(MSAModel, self).__init__()
        self.proj_dim, self.orig_dim, self.num_mod, self.output_dim = proj_dim, orig_dim, len(orig_dim), output_dim
        self.relu_dropout, self.embed_dropout, self.res_dropout, self.out_dropout, self.attn_dropout = \
            relu_dropout, embed_dropout, res_dropout, out_dropout, attn_dropout

        self.input_norms = nn.ModuleList([nn.LayerNorm(self.orig_dim[i]) for i in range(self.num_mod)])
        self.proj = nn.ModuleList([nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0) for i in range(self.num_mod)])
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim=proj_dim, num_heads=num_heads, layers=layers, attn_dropout=attn_dropout,
                               res_dropout=res_dropout, relu_dropout=relu_dropout, embed_dropout=embed_dropout)
            for _ in range(self.num_mod)
        ])

        self.gate_classifiers = UnimodalAuxiliaryHeads(output_dim, self.num_mod, proj_dim)

        # Learnable gating mechanism (replaces the detached no_grad gating)
        self.gate_attention = GateAttention(proj_dim, self.num_mod, temperature=1.0)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.proj_dim * self.num_mod, self.proj_dim * 2),
            nn.ReLU(), nn.Dropout(p=self.out_dropout), nn.Linear(self.proj_dim * 2, self.proj_dim)
        )
        self.proj1, self.proj2, self.out_layer = \
            nn.Linear(self.proj_dim, self.proj_dim), nn.Linear(self.proj_dim, self.proj_dim), nn.Linear(self.proj_dim, self.output_dim)

        # Dedicated unimodal heads for ablation study
        self.unimodal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.proj_dim, self.proj_dim),
                nn.ReLU(),
                nn.Linear(self.proj_dim, self.output_dim)
            ) for _ in range(self.num_mod)
        ])

    def forward(self, x, unimodal_mode=None):
        # Apply LayerNorm before projection
        x_normed = [self.input_norms[i](x[i]) for i in range(self.num_mod)]
        hs = [self.encoders[i](self.proj[i](x_normed[i].transpose(1, 2)).permute(2, 0, 1)) for i in range(self.num_mod)]

        # Evaluation-only path for ablation
        if unimodal_mode is not None:
            cls_token = hs[unimodal_mode][0]
            unimodal_output = self.unimodal_heads[unimodal_mode](cls_token)
            return {'unimodal_output': unimodal_output}

        # Multimodal fusion path
        h_cls_tokens = [h[0] for h in hs]

        # Auxiliary unimodal classifiers (for gate loss / OGM)
        unimodal_logits = self.gate_classifiers(h_cls_tokens)

        # LEARNABLE gating: gradients flow through to encoders
        gates = self.gate_attention(h_cls_tokens)  # [B, num_mod]

        # Gated fusion input
        gated_cls = []
        for i in range(self.num_mod):
            gated_cls.append(h_cls_tokens[i] * gates[:, i].unsqueeze(1))
        fused_input = torch.cat(gated_cls, dim=1)  # [B, proj_dim * num_mod]
        fused_vector = self.fusion_mlp(fused_input)

        # MODALITY-AGNOSTIC fusion: no text residual shortcut
        last_hs = fused_vector

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        return {
            "output": output, "unimodal_logits": unimodal_logits,
            "hs_nondetached": hs, "hs_detached": [h.clone().detach() for h in hs],
            "gates": gates  # expose for logging/visualization
        }