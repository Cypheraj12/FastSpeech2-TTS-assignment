"""
it is the Re-implementation that follows the structure of the reference repo
(ming024/FastSpeech2): FFT-block encoder -> variance adaptor (duration / pitch /
energy predictors + length regulator) -> FFT-block decoder -> mel linear -> PostNet.
Pitch and energy are phoneme(token)-level, quantised into 256 bins and embedded.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    """True where the position is padding."""
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def sinusoid_table(n_pos, d, device):
    pos = torch.arange(n_pos, device=device).unsqueeze(1).float()
    i = torch.arange(d, device=device).unsqueeze(0).float()
    angle = pos / torch.pow(10000.0, 2 * (i // 2) / d)
    pe = torch.zeros(n_pos, d, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class FFTBlock(nn.Module):
    """Self-attention + 2-layer 1D-conv feed-forward (kernel 9 and 1), post-LN."""

    def __init__(self, d_model, n_heads, d_inner, kernel_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_inner, kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.conv2 = nn.Conv1d(d_inner, d_model, kernel_size[1], padding=(kernel_size[1] - 1) // 2)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.ln1(x + self.drop(a))
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        y = self.conv2(self.drop(F.relu(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
        x = self.ln2(x + self.drop(y))
        return x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class VariancePredictor(nn.Module):
    """2 x (Conv1d -> ReLU -> LayerNorm -> Dropout) -> Linear."""

    def __init__(self, d_model, filt, kernel, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filt, kernel, padding=(kernel - 1) // 2)
        self.ln1 = nn.LayerNorm(filt)
        self.conv2 = nn.Conv1d(filt, filt, kernel, padding=(kernel - 1) // 2)
        self.ln2 = nn.LayerNorm(filt)
        self.lin = nn.Linear(filt, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = F.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        y = self.drop(self.ln1(y))
        y = F.relu(self.conv2(y.transpose(1, 2))).transpose(1, 2)
        y = self.drop(self.ln2(y))
        return self.lin(y).squeeze(-1).masked_fill(pad_mask, 0.0)


def length_regulate(x, durations):
    """Repeat each token vector `durations[b, i]` times. Returns (B,T,D), lengths."""
    outs = [torch.repeat_interleave(x[b], durations[b].long(), dim=0) for b in range(x.size(0))]
    lens = torch.tensor([o.size(0) for o in outs], device=x.device)
    return nn.utils.rnn.pad_sequence(outs, batch_first=True), lens


class VarianceAdaptor(nn.Module):
    def __init__(self, m, pitch_range, energy_range):
        super().__init__()
        d, nb = m["d_model"], m["n_bins"]
        mk = lambda: VariancePredictor(d, m["var_filter"], m["var_kernel"], m["var_dropout"])
        self.duration_predictor, self.pitch_predictor, self.energy_predictor = mk(), mk(), mk()
        self.register_buffer("pitch_bins", torch.linspace(pitch_range[0], pitch_range[1], nb - 1))
        self.register_buffer("energy_bins", torch.linspace(energy_range[0], energy_range[1], nb - 1))
        self.pitch_embedding = nn.Embedding(nb, d)
        self.energy_embedding = nn.Embedding(nb, d)

    def forward(self, x, src_mask, d_target=None, p_target=None, e_target=None,
                d_control=1.0, p_control=1.0, e_control=1.0):
        log_d = self.duration_predictor(x, src_mask)          # predicts log(d + 1)
        p_pred = self.pitch_predictor(x, src_mask)
        p = p_target if p_target is not None else p_pred * p_control
        x = x + self.pitch_embedding(torch.bucketize(p, self.pitch_bins))
        e_pred = self.energy_predictor(x, src_mask)
        e = e_target if e_target is not None else e_pred * e_control
        x = x + self.energy_embedding(torch.bucketize(e, self.energy_bins))
        if d_target is not None:
            d_rounded = d_target
        else:
            d_rounded = torch.clamp(torch.round(torch.exp(log_d.float()) - 1.0) * d_control, min=0)
            d_rounded = d_rounded.masked_fill(src_mask, 0)
            empty = d_rounded.sum(1) == 0                      # avoid zero-length output
            if empty.any():
                d_rounded[empty] = (~src_mask[empty]).float()
        x, mel_len = length_regulate(x, d_rounded)
        return x, p_pred, e_pred, log_d, d_rounded, mel_len


class PostNet(nn.Module):
    def __init__(self, n_mels=80, dim=512, n_conv=5, kernel=5):
        super().__init__()
        layers = []
        for i in range(n_conv):
            cin = n_mels if i == 0 else dim
            cout = n_mels if i == n_conv - 1 else dim
            block = [nn.Conv1d(cin, cout, kernel, padding=(kernel - 1) // 2), nn.BatchNorm1d(cout)]
            if i < n_conv - 1:
                block += [nn.Tanh(), nn.Dropout(0.5)]
            else:
                block += [nn.Dropout(0.5)]
            layers.append(nn.Sequential(*block))
        self.convs = nn.ModuleList(layers)

    def forward(self, x):                                      # x: (B,T,n_mels)
        y = x.transpose(1, 2)
        for c in self.convs:
            y = c(y)
        return y.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg, n_symbols, stats, n_speakers=1):
        super().__init__()
        m, self.n_mels = cfg["model"], cfg["audio"]["n_mels"]
        d = m["d_model"]
        self.d_model = d
        self.embedding = nn.Embedding(n_symbols, d, padding_idx=0)
        mk = lambda: FFTBlock(d, m["n_heads"], m["ff_dim"], m["kernel_size"], m["dropout"])
        self.encoder = nn.ModuleList([mk() for _ in range(m["enc_layers"])])
        self.variance_adaptor = VarianceAdaptor(
            m, (stats["pitch_min"], stats["pitch_max"]), (stats["energy_min"], stats["energy_max"]))
        self.decoder = nn.ModuleList([mk() for _ in range(m["dec_layers"])])
        self.mel_linear = nn.Linear(d, self.n_mels)
        self.postnet = PostNet(self.n_mels)
        # multi-speaker conditioning (same idea as the reference repo): a learned speaker vector is
        # added to every encoder output frame. Only created when the corpus has >1 speaker.
        self.speaker_emb = nn.Embedding(n_speakers, d) if n_speakers > 1 else None

    def forward(self, src, src_len, speaker=None, d_target=None, p_target=None, e_target=None,
                d_control=1.0, p_control=1.0, e_control=1.0):
        src_mask = get_mask_from_lengths(src_len, src.size(1))
        x = self.embedding(src) + sinusoid_table(src.size(1), self.d_model, src.device)
        for blk in self.encoder:
            x = blk(x, src_mask)
        if self.speaker_emb is not None and speaker is not None:
            x = x + self.speaker_emb(speaker).unsqueeze(1)
        x, p_pred, e_pred, log_d, d_rounded, out_len = self.variance_adaptor(
            x, src_mask, d_target, p_target, e_target, d_control, p_control, e_control)
        mel_mask = get_mask_from_lengths(out_len, x.size(1))
        x = x + sinusoid_table(x.size(1), self.d_model, x.device)
        x = x.masked_fill(mel_mask.unsqueeze(-1), 0.0)
        for blk in self.decoder:
            x = blk(x, mel_mask)
        mel = self.mel_linear(x)
        mel_post = mel + self.postnet(mel)
        return dict(mel=mel, mel_post=mel_post, p_pred=p_pred, e_pred=e_pred, log_d_pred=log_d,
                    d_rounded=d_rounded, src_mask=src_mask, mel_mask=mel_mask, mel_len=out_len)


def fastspeech2_loss(out, mel_t, d_t, p_t, e_t):
    """L1 on mel (before and after PostNet) + MSE on log-duration, pitch, energy."""
    keep_mel = (~out["mel_mask"]).float()
    keep_src = (~out["src_mask"]).float()
    T = out["mel"].size(1)
    mel_t = mel_t[:, :T]
    n_mel = mel_t.size(-1)

    def l1(pred):
        return ((pred.float() - mel_t).abs() * keep_mel.unsqueeze(-1)).sum() / (keep_mel.sum() * n_mel)

    def mse(pred, tgt):
        return (((pred.float() - tgt) ** 2) * keep_src).sum() / keep_src.sum()

    mel_loss, post_loss = l1(out["mel"]), l1(out["mel_post"])
    dur_loss = mse(out["log_d_pred"], torch.log(d_t.float() + 1.0))
    pitch_loss, energy_loss = mse(out["p_pred"], p_t), mse(out["e_pred"], e_t)
    total = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss
    return dict(total=total, mel=mel_loss, postnet=post_loss, duration=dur_loss,
                pitch=pitch_loss, energy=energy_loss)
