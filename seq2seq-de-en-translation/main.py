import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, Iterator
from torchtext import datasets
import numpy as np

import time
import random
from pathlib import Path

from config import *
from model import Encoder, Decoder, Attention, Seq2Seq
from utils import init_weights, epoch_time, translate_sentence


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        er = loss.item()
        epoch_loss += er
        if _ % 100 == 0:
            print(f'Num iter: {_} / {len(iterator)}', f'\t Cur train loss: {er:.3f}', flush=True)

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)  # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def run():
    Path('models').mkdir(parents=True, exist_ok=True)

    val_data_de = []
    with open(DATA_DIR + 'val.de-en.de') as f:
        for l in f:
            val_data_de.append(l.split('\n')[0])

    train_data_de = []
    with open(DATA_DIR + 'train.de-en.de') as f:
        for l in f:
            train_data_de.append(l.split('\n')[0])

    val_data_en = []
    with open(DATA_DIR + 'val.de-en.en') as f:
        for l in f:
            val_data_en.append(l.split('\n')[0])

    train_data_en = []
    with open(DATA_DIR + 'train.de-en.en') as f:
        for l in f:
            train_data_en.append(l.split('\n')[0])

    with open(DATA_DIR + 'my_train.en', 'w') as f:
        for s in train_data_en:
            f.write(s + '\n')
        for s in val_data_en:
            f.write(s + '\n')

    with open(DATA_DIR + 'my_train.de', 'w') as f:
        for s in train_data_de:
            f.write(s + '\n')
        for s in val_data_de:
            f.write(s + '\n')

    SRC = Field(lower=True, init_token="<sos>", eos_token="<eos>")

    TRG = Field(lower=True, init_token="<sos>", eos_token="<eos>")

    train_data = datasets.TranslationDataset(
        path=DATA_DIR + 'my_train', exts=('.de', '.en'),
        fields=(SRC, TRG))

    val_data = datasets.TranslationDataset(
        path=DATA_DIR + 'val.de-en', exts=('.de', '.en'),
        fields=(SRC, TRG))

    test_data = datasets.TranslationDataset(
        path=DATA_DIR + 'test1.de-en', exts=('.de', '.de'),
        fields=(SRC, SRC))

    SRC.build_vocab(train_data, min_freq=3)
    TRG.build_vocab(train_data, min_freq=3)

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, val_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device,
                         sort=False, sort_within_batch=False, repeat=False)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
    model.apply(init_weights)

    # print(f'The model has {count_parameters(model):,} trainable parameters')
    LR = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    PAD_IDX = TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    N_EPOCHS = 4
    CLIP = 1

    best_valid_loss = float('inf')
    torch.cuda.empty_cache()

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        if epoch % 2 == 0 and epoch != 0:
            LR /= 2
            optimizer = optim.AdamW(model.parameters(), lr=LR)

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # torch.save(model.state_dict(), f'models/tt_model_{epoch}.pt')

        example_idx = 18

        src = vars(train_data.examples[example_idx])['src']
        trg = vars(train_data.examples[example_idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        translation = translate_sentence(' '.join(src), SRC, TRG, model, device)

        print(f'predicted trg = {translation} \n')
        # bleu_score = calculate_bleu(train_data, SRC, TRG, model, device)

        # print(f'BLEU score = {bleu_score*100:.2f}')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    test1 = []
    with open(DATA_DIR + 'test1.de-en.de') as f:
        for l in f:
            test1.append(l.split('\n')[0])

    res = []
    for idx, i in enumerate(test1):
        model.eval()
        translated_sentence = translate_sentence(
            i, SRC, TRG, model, device, 100)
        res.append(translated_sentence)

    with open(DATA_DIR + 'test1.de-en.en', 'w') as f:
        for s in res:
            f.write(' '.join(s[:-1]) + '\n')

    data_en = []
    with open(DATA_DIR + 'test1.de-en.en') as f:
        for l in f:
            data_en.append(l.split('\n')[0].split())

    res = []
    for data in data_en:
        j = 0
        new_s = []
        if len(data) <= 2:
            res.append(data)
            continue
        while j < len(data) - 1:
            if data[j + 1] == data[j]:
                new_s.append(data[j])
                j += 1
            else:
                new_s.append(data[j])
            j += 1
        new_s.append(data[-1])
        res.append(new_s)
    with open(DATA_DIR + 'test1.de-en.en', 'w') as f:
        for s in res:
            f.write(' '.join(s) + '\n')


if __name__ == '__main__':
    run()
