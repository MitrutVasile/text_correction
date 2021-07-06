import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import padding, save_checkpoint_transformer, correct_sentence_transformer, load_checkpoint

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

with open('soruce.txt', 'r', encoding='utf-8') as f:
    #propozitii_eronate = f.readlines()
    propozitii_eronate = [next(f) for x in range(2000000)]

with open('target.txt', 'r', encoding='utf-8') as ff:
    #propozitii_corecte = ff.readlines()
    propozitii_corecte = [next(ff) for p in range(2000000)]

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True

save_model = False

# Training hyperparameters
num_epochs = 10000
learning_rate = 0.0001


# Model hyperparameters
src_vocab_size = 8000
trg_vocab_size = 8000
embedding_size = 1024
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 150
forward_expansion = 4
src_pad_idx = 0


writer = SummaryWriter("runs/loss_plot")
step = 0


model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


'''
pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
import sys
if load_model:
    print(load_checkpoint(torch.load("checkpointtrasformer.pth.tar"), model, optimizer))
    load_checkpoint(torch.load("checkpointtrasformer.pth.tar"), model, optimizer)
#sentence = 'Cnsideer că cera ce ne fdefinește cla oampni esrte mintea, creeiierul urmman'

sentence = 'mamaea eiste la noi acas'
model.eval()
print(correct_sentence_transformer(model, sentence,device, max_length = 150 ))
sys.exit()
lenn = 32
propozitii = propozitii_corecte.copy()
prop_eron = propozitii_eronate.copy()
for epoch in range(num_epochs):


    print(f'Epoch [{epoch} / {num_epochs}]')
    losses = []
    #model.eval()
    #propozitie_corectata = correct_sentence_transformer(model, sentence, device, max_length = 100)
    #print(propozitie_corectata)
    #model.train()

    if save_model:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint_transformer(checkpoint)

    for i in range(len(propozitii)):
        input_data = torch.tensor([])
        target_data = torch.tensor([])
        if i % lenn == 0:
            #print("propozitia", str(i), "din", str(len(propozitii)))
            propozitii_corecte = propozitii[i: i + lenn]

            propozitii_corecte = list(map(lambda s: s.strip(), propozitii_corecte))
            propozitii_eronate = prop_eron[i: i + lenn]

            tensor_correct = padding(propozitii_corecte, True, lenn)
            tensor_error = padding(propozitii_eronate, True, lenn)


            input_data = torch.cat((input_data, tensor_error), 1)
            input_data = input_data.long().to(device)
            #print(input_data)
            #print(input_data.size())
            target_data = torch.cat((target_data, tensor_correct), 1)
            target_data = target_data.long().to(device)
            #print(target_data)
            #print(target_data[:-1,:].size())

            output = model(input_data, target_data[:-1,:])
            # print('trecerea prin model')
            output = output.reshape(-1, output.shape[2])
            target_data = target_data[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target_data)
            #print(loss.item())
            losses.append(loss.item())
            loss.backward()
            # print('calcul loss')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            writer.add_scalar('Training loss', loss, global_step=step)
            step += 1

    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')
'''