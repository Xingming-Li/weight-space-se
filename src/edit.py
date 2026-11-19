# Weight edit: remove a direction d (language difference) from what the model writes into its residual stream.
# W' = W - alpha * d d^T W, no training or gradients
# 

import torch


def edit_targets(model):
    yield "embed", model.model.embed_tokens.weight
    for i, layer in enumerate(model.model.layers):
        yield f"o_proj.{i}", layer.self_attn.o_proj.weight
        yield f"down_proj.{i}", layer.mlp.down_proj.weight


# Editing the embedding should not change the readout
# Snapshot the weights that will be edited
@torch.no_grad()
def untie_and_snapshot(model):
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
        print("Untied lm_head from embeddings")
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    return {n: p.detach().cpu().clone() for n, p in edit_targets(model)}


@torch.no_grad()
def restore(model, snap):
    for n, p in edit_targets(model):
        p.copy_(snap[n].to(p.device, p.dtype))


@torch.no_grad()
def project_out(model, d, alpha=1.0, chunk=16384):
    d = d.float().to(model.device)
    d = d / d.norm()
    for name, p in edit_targets(model):
        if name == "embed":  # Rows live in the residual space
            for s in range(0, p.shape[0], chunk):
                blk = p[s:s + chunk].float()
                blk -= alpha * (blk @ d)[:, None] * d[None, :]
                p[s:s + chunk].copy_(blk.to(p.dtype))
        else:  # [hidden, in]: remove d from outputs
            W = p.float()
            W -= alpha * d[:, None] * (d @ W)[None, :]
            p.copy_(W.to(p.dtype))


# Per-text mean residual stream vector at each requested layer.
@torch.no_grad()
def text_means(model, tok, texts, layers, batch_size=8, max_len=512):
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        attn = enc.attention_mask.bool()
        # First real token of is excluded
        first = (enc.attention_mask.cumsum(1) == 1) & attn
        m = (attn & ~first).unsqueeze(-1).float()
        per_layer = []
        for L in layers:
            h = hs[L].float()
            per_layer.append((h * m).sum(1) / m.sum(1))  # [n_texts, hidden]
        out.append(torch.stack(per_layer, 1).cpu())  # [n_texts, n_layers, hidden]
    return torch.cat(out)  


def language_direction(sv_means, en_means):
    return (sv_means - en_means).mean(0)


# Randomly flip the sign of each paired difference
def null_direction(sv_means, en_means, gen):
    delta = sv_means - en_means
    signs = (torch.randint(0, 2, (delta.shape[0], 1, 1), generator=gen) * 2 - 1).float()
    return (delta * signs).mean(0)