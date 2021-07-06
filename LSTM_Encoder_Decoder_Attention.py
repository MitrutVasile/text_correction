import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
torch.manual_seed(0)

with open('soruce.txt', 'r', encoding='utf-8') as f:
    #propozitii_eronate = [next(f) for x in range(3000000)]
    propozitii_eronate = f.readlines()
#print(len(propozitii_eronate))
with open('target.txt', 'r', encoding='utf-8') as ff:
    #propozitii_corecte = [next(ff) for x in range(3000000)]
    propozitii_corecte = ff.readlines()
#print(len(propozitii_corecte))


print(torch.cuda.is_available())

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # forma x: (seq_length, N)  N reprezinta dimensiunea lotului

        embedding = self.dropout(self.embedding(x))
        # forma embedding: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding) #hidden, cell - vectorul de context
        # forma output: (seq_length, N, hidden_size)


        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N)  N - dimensiunea lotului
        embedding = self.dropout(self.embedding(x))
        # embedding: (1, N, embedding_size)
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)
        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)
        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # context_vector: (1, N, hidden_size*2)  knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs: (1, N, hidden_size)
        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=1):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = 8000

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        x = target[0] #tokenul de inceput de propozitie

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs



#device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False


# Training hyperparameters
num_epochs = 1000
learning_rate = 0.0001
batch_size = 16

# Model hyperparameters
input_size_encoder = 8000
input_size_decoder = 8000
output_size = 8000
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 512

num_layers = 1
enc_dropout = 0.1
dec_dropout = 0.1

# plotare loss
writer = SummaryWriter(f"runs/loss_plot2")
step = 0


encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
'''
pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
lenn = 64
if load_model:
    load_checkpoint(torch.load("checkpointlstmatt2.pth.tar"))


sentence = 'Teritorii dependante și vecinii autonome în Euopa.'


propozitii = propozitii_corecte.copy()
prop_eron = propozitii_eronate.copy()

for epoch in range(num_epochs):


    print(f'Epoch [{epoch} / {num_epochs}]')
    losses = []
    #model.eval()
    #propozitie_corectata = correct_sentence(model, sentence, device, max_length = len(tokenizer.encode(sentence, add_special_tokens=False).ids))
    #print(propozitie_corectata)
    #model.train()


    for i in range(len(propozitii)):

        input_data = torch.tensor([])
        target_data = torch.tensor([])
        if i % lenn == 0:
            print("propozitia", str(i), "din", str(len(propozitii)))
            propozitii_corecte = propozitii[i: i + lenn]  #selectarea mini-batch-ului pentru target_data
            propozitii_corecte = list(map(lambda s: s.strip(), propozitii_corecte))
            propozitii_eronate = prop_eron[i: i + lenn] #selectarea mini-batch-ului pentru input_data
            #padding
            tensor_correct = padding(propozitii_corecte, True, lenn)
            tensor_error = padding(propozitii_eronate, True, lenn)
            #datele de intrare respectiv ieșire
            input_data = torch.cat((input_data, tensor_error), 1)
            input_data = input_data.long().to(device)
            target_data = torch.cat((target_data, tensor_correct), 1)
            target_data = target_data.long().to(device)
            #trecerea prin model
            output = model(input_data, target_data)
            #modificarea dimensiunilor pentru calcului Loss-ului
            output = output[1:].reshape(-1, output.shape[2])
            target_data = target_data[1:].reshape(-1)
            #reinitializarea gradientilor cu zerouri
            optimizer.zero_grad()
            #calculul loss-ului pentru fiecare batch
            loss = criterion(output,target_data)
            print(loss.item())
            losses.append(loss.item())
            #backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()
            #if step % 10 == 0:
            #    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #    save_checkpoint(checkpoint)
            writer.add_scalar('Training loss', loss, global_step=step)
            step += 1
            print(step)

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')
    '''