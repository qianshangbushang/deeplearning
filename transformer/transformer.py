import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        assert (self.head_dim * self.heads ==
                self.emb_size), "emb size should be div by heads"
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.heads * self.head_dim, self.emb_size)
        return

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values, keys, queries = self.values(values), self.keys(keys), self.queries(queries)

        # QK
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # nhqk
        attention = torch.softmax(energy/(self.emb_size ** (1/2)), dim=3)

        out = torch.einsum("nhqk,hkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class TransformBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout) -> None:
        super().__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size,  emb_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(emb_size * forward_expansion, emb_size),
        )

        self.dropout = nn.Dropout(dropout)
        return

    def feed_forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + queries))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class Encoder(nn.Module):
    def __init__(
            self, 
            vocab_size,
            emb_size,
            device,
            heads,
            num_layers,
            forward_expansion,
            dropout,
            max_length,
            ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_length, emb_size)

        self.layers = nn.ModuleList([
            TransformBlock(
                emb_size=emb_size,
                heads=heads,
                forward_expansion=forward_expansion,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        return
    
    def forward(self, x, mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out=self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out,mask)
        return out
    


class DecodeBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout) -> None:
        super().__init__()
        self.attenion = SelfAttention(emb_size, heads)
        self.norm = nn.LayerNorm(emb_size)
        self.transform_block = TransformBlock(emb_size, heads, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        return
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attenion(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out=self.transform_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_length, emb_size, heads, forward_expansion, dropout, num_layers, device) -> None:
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_length, emb_size)
        self.layers = nn.ModuleList([
            DecodeBlock(emb_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout= nn.Dropout(dropout)
        return
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        query = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            query = layer(query, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(query)
    


class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion = 4,
            heads = 8,
            dropout=0,
            device='cuda',
            max_length=100,
            ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, device, heads, num_layers, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, max_length, embed_size, heads, forward_expansion, dropout, num_layers, device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        return
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out