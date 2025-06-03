import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiStreamKeyPredictor(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_classes=12, dropout=0.3):
        super().__init__()

        # Chroma Stream
        self.chroma_proj = nn.Linear(12, d_model)
        self.chroma_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # HPCP Stream
        self.hpcp_proj = nn.Linear(12, d_model)
        self.hpcp_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Fusion + Attention Pooling
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.fc_out = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, feats, src_key_padding_mask=None):
        chroma_feats = feats[:, :, :12]
        hpcp_feats = feats[:, :, 12:]

        chroma_emb = self.chroma_proj(chroma_feats)
        hpcp_emb = self.hpcp_proj(hpcp_feats)

        chroma_out = self.chroma_encoder(chroma_emb, src_key_padding_mask=src_key_padding_mask)
        hpcp_out = self.hpcp_encoder(hpcp_emb, src_key_padding_mask=src_key_padding_mask)

        fused = torch.cat([chroma_out, hpcp_out], dim=-1)
        fused = self.fusion_fc(fused)

        B, T, D = fused.shape
        cls_token = self.cls_token.expand(B, 1, D)
        fused = torch.cat([cls_token, fused], dim=1)  # [B, T+1, D]

        # Key padding mask 처리
        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            cls_padding = torch.zeros((B, 1), dtype=torch.bool, device=src_key_padding_mask.device)
            attn_mask = torch.cat([cls_padding, src_key_padding_mask], dim=1)
        else:
            attn_mask = None

        attn_out, _ = self.attention(fused, fused, fused, key_padding_mask=attn_mask)

        pooled = attn_out[:, 0, :]  # [B, D]

        # Normalize & Linear projection
        pooled = F.normalize(pooled.float(), dim=-1).to(pooled.dtype)
        logits = F.linear(pooled, F.normalize(self.fc_out.weight.float(), dim=-1)).to(pooled.dtype)

        return logits
