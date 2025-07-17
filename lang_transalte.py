# model.py
import torch
from torch import nn
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
print("imported modules")

# === PARAMS ===
d_model = 64
num_heads = 4
drop_prob = 0.1
max_sequence_length = 10
ffn_hidden = 128
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD & CLEAN DATA ===
dataset = pd.read_csv('trans.csv', header=None)
hin = dataset[0].astype(str).tolist()
eng = dataset[1].astype(str).tolist()

# Force add <start> <end> if missing
if not hin[0].startswith("<start>"):
    hin = [f"<start> {x.strip()} <end>" for x in hin]

# === TOKENIZE & PAD ===
eng_token = Tokenizer(filters='')
eng_token.fit_on_texts(eng)
eng_seq = eng_token.texts_to_sequences(eng)
eng_seq = pad_sequences(eng_seq, maxlen=max_sequence_length, padding='post')

hin_token = Tokenizer(filters='')
hin_token.fit_on_texts(hin)
hin_seq = hin_token.texts_to_sequences(hin)
hin_seq = pad_sequences(hin_seq, maxlen=max_sequence_length, padding='post')

src_vocab = len(eng_token.word_index) + 1
tgt_vocab = len(hin_token.word_index) + 1

# === MODEL PARTS ===
emb_enc = nn.Embedding(src_vocab, d_model).to(device)
emb_dec = nn.Embedding(tgt_vocab, d_model).to(device)
pos_enc = nn.Embedding(max_sequence_length, d_model).to(device)
pos_dec = nn.Embedding(max_sequence_length, d_model).to(device)

mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(device)
self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(device)
cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(device)
dropout = nn.Dropout(drop_prob).to(device)

ffn = nn.Sequential(
    nn.Linear(d_model, ffn_hidden),
    nn.ReLU(),
    nn.Dropout(drop_prob),
    nn.Linear(ffn_hidden, d_model)
).to(device)

layernorms_enc = [nn.LayerNorm(d_model).to(device) for _ in range(num_layers * 2)]
layernorms_dec = [nn.LayerNorm(d_model).to(device) for _ in range(num_layers * 3)]

# === MASK ===
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

# === ENCODER ===
def encoder_layer(x, ln1, ln2):
    res = x
    attn, _ = mha(x, x, x)
    x = ln1(res + dropout(attn))
    x = ln2(x + dropout(ffn(x)))
    return x

# === DECODER ===
def decoder_layer(enc, dec, ln1, ln2, ln3, mask):
    res = dec
    attn1, _ = self_attn(dec, dec, dec, attn_mask=mask)
    dec = ln1(res + dropout(attn1))

    res = dec
    attn2, _ = cross_attn(dec, enc, enc)
    dec = ln2(res + dropout(attn2))

    res = dec
    dec = ln3(res + dropout(ffn(dec)))
    return dec

# === TRANSLATE ===
def translate(input_sentence):
    seq = eng_token.texts_to_sequences([input_sentence])
    seq = pad_sequences(seq, maxlen=max_sequence_length, padding='post')
    src = torch.tensor(seq).to(device)

    src_emb = emb_enc(src) + pos_enc(torch.arange(max_sequence_length).to(device))
    enc = src_emb
    for i in range(num_layers):
        enc = encoder_layer(enc, layernorms_enc[i*2], layernorms_enc[i*2+1])

    start_token = hin_token.word_index.get('<start>', 1)
    end_token = hin_token.word_index.get('<end>', 2)
    tgt_seq = [start_token]

    for _ in range(max_sequence_length):
        tgt = torch.tensor([tgt_seq + [0]*(max_sequence_length - len(tgt_seq))]).to(device)
        tgt_emb = emb_dec(tgt) + pos_dec(torch.arange(max_sequence_length).to(device))

        dec = tgt_emb
        mask = generate_square_subsequent_mask(max_sequence_length)
        for i in range(num_layers):
            dec = decoder_layer(enc, dec, layernorms_dec[i*3], layernorms_dec[i*3+1], layernorms_dec[i*3+2], mask)

        next_token = dec[0, len(tgt_seq)-1].argmax(-1).item()
        tgt_seq.append(next_token)
        if next_token == end_token:
            break

    idx2word = {v: k for k, v in hin_token.word_index.items()}
    output = [idx2word.get(idx, '') for idx in tgt_seq[1:] if idx not in [0, start_token, end_token]]
    return ' '.join(output)
