import torch
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
def returnMaxLen(sentences, special_tokens):

    input_ids = torch.tensor(tokenizer.encode(sentences[0], add_special_tokens=special_tokens).ids).unsqueeze(0)
    max_len = input_ids.size()[1]

    for i in range(1, len(sentences)):
        input_ids1= torch.tensor(tokenizer.encode(sentences[i], add_special_tokens = special_tokens).ids).unsqueeze(0)

        if input_ids1.size()[1] >= max_len:
            max_len = input_ids1.size()[1]

    return max_len
def padding(propozitii, special_tokens, length):
    maxx_len = returnMaxLen(propozitii, special_tokens) # calcularea dimensiunii celei mai lungi propozitii

    sent_tensor = torch.zeros((length, maxx_len))
    for i in range(length):
        #tokenizarea propozitiei initiale
        initial_sentance = torch.tensor(tokenizer.encode(propozitii[i], add_special_tokens=special_tokens).ids).unsqueeze(0)
        #padding-ul pentru fiecare propozitie
        sentance_padding = torch.stack([torch.cat([i, i.new_zeros(maxx_len - i.size(0))], 0) for i in initial_sentance],1)

        sent_tensor[i] = torch.reshape(sentance_padding, (1, sentance_padding.size()[0]))

    return torch.transpose(sent_tensor,0,1)

def save_checkpoint(state, filename = "checkpointlstmatt_new.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
def save_checkpoint_transformer(state, filename = "checkpointtrasformer.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
def correct_sentence(model, sentence, device, max_length):

    sentence_tensor = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True).ids).unsqueeze(1).long().to(device)

    with torch.no_grad():
        outputs_encoder, hidden, cell = model.encoder(sentence_tensor)
    outputs = [2]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word,outputs_encoder, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        #print("predictia: ", outputs)
        if output.argmax(1).item() == 3:
            break

    corrected_sentence = tokenizer.decode(outputs, skip_special_tokens=True)


    return corrected_sentence

def correct_sentence_transformer(model, sentence, device, max_length):
    #codarea propoziției folosind tokenizer-ul
    sentence_tensor = torch.LongTensor(tokenizer.encode(sentence,
                                                        add_special_tokens=True).ids).unsqueeze(1).to(device)
    #tokenul de inceput pentru predictia ce va fi făcută
    outputs = [2]
    for _ in range(max_length):

        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        #print(trg_tensor)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)
        #print("predictia: ", outputs)
        #execută cât timp predicția este diferită de 3 (tokenul de final)
        if best_guess == 3:
            break
    #decodarea secvenței obținute
    translated_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
    return translated_sentence