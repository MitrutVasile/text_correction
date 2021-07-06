from Levenshtein import distance as levenshtein_distance
import numpy as np
import jiwer
import random
from collections.abc import Iterable
import torch
from sortareFrecventaCuvinte import text2
from histograma_litere import dict_litere
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

mean = 0.4
abatere_vector = np.linspace(0.1, 0.15, 70)
pozitie = int((mean*100)-10)
abatere = abatere_vector[pozitie]
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#obtinerea unei liste unice dintr-o lista care contine alte liste ca elemente
def flatten(list1):
    for el in list1:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

#transpunerea unei liste intr-un string
def listToString(s):
    strr = " "
    return strr.join(s)

# inserarea unei litere

#inserarea unei litere aleatoare
def inserare_litera(word):

    index = random.randint(1,len(word))
    cuvant_eronat = word[:index] + random.choice(dict_litere) + word[index:]
    return cuvant_eronat

# inlocuirea unei litere

def inlocuire_litera(word):

    if len(word) > 2:
        inds = [i for i, _ in enumerate(word) if not word.isspace()]
        sam = random.sample(inds, 1) #pozitia unde va fi introdusa litera
        lst = list(word)
        for ind in sam:
            lst[ind] = random.choice(dict_litere)

        return "".join(lst)
    else:
        return word

# inlocuirea unui cuvant

def inlocuire_cuvant(word):

    l=[]

    for x in text2: # text2 reprezintă lista de cuvinte
        if levenshtein_distance(word, x) == random.randint(2,4): #alegerea random a distantei levenshtein
            l.append(x)

    if not l: # in cazul in care nu se gaseste cuvant de inlocuit returnam acelasi cuvant

        return word
    else:

        while True:
            y = random.choice(l)
            if y.lower()!= word.lower():
                #print("cuvantul cu care s-a inlocuit:", y)
                break
            else:
                #print('nu s-a gasit niciun cuvant de inlocuit')
                y = word
                break

    return y

def eliminare_litera(word):

    r = random.randrange(len(word))
    return word[:r] + word[r + 1:]

def eronare_subcuvant(word):

    a = random.randint(2,random.randint(3,len(word)))
    return inlocuire_cuvant(word[0:a]), inlocuire_cuvant(word[a:len(word)])


def eronare_text(text,lista_cuvinte_eronate):

    while True:
        i = random.choice(text)
        if str(text.index(i)) not in lista_cuvinte_eronate and len(i) > 2:
            break
        #elif len(i) != 0 and i not in lista_cuvinte_eronate:
        elif len(i) != 0 and str(text.index(i)) not in lista_cuvinte_eronate:
            #print("\n s- a facut inserare litera\n")
            cuvant_eronat = inserare_litera(i)
            text[text.index(i)] = cuvant_eronat
            return text, text.index(cuvant_eronat)
    #print("cuvantul care va fi eronat:", i)
    p = random.randint(0, 4) #alegerea random a metodei de eronare

    if p == 0:

        cuvant_eronat = inlocuire_litera(i)
        text[text.index(i)] = cuvant_eronat
        return text, text.index(cuvant_eronat)
    elif p == 1:

        cuvant_eronat = inserare_litera(i)
        text[text.index(i)] = cuvant_eronat
        return text, text.index(cuvant_eronat)
    elif p == 2:

        cuvant_eronat = eliminare_litera(i)
        text[text.index(i)] = cuvant_eronat
        return text, text.index(cuvant_eronat)
    elif p == 3:
        if len(i)>4:

            cuvant_eronat = inlocuire_cuvant(i)
            text[text.index(i)] = cuvant_eronat
            return text, text.index(cuvant_eronat)
        else:


            return text, text.index(i)
    elif p == 4 :
        if len(i) > 6:
            cuvant_eronat = text[text.index(i)]
            x = eronare_subcuvant(text[text.index(i)])
            text[text.index(i)] = list(x)

            text = list(flatten(text))

            return text, text.index(list(x)[0])
        else:
            cuvant_eronat = text[text.index(i)]
            return text, text.index(cuvant_eronat)







def eronare_text2(propozitii):
    numar_propozitii = len(propozitii)
    x_values = np.linspace(mean - abatere, mean + abatere, numar_propozitii)
    propozitii_eronate = propozitii.copy()
    k = 1
    tuplu_eronat = ()
    for i in range(len(propozitii)):
        t = []
        print('s-a eronat propozitia: ', k, 'din', len(propozitii))
        rata_impusa_eronare = round(random.choice(x_values), 2)
        k += 1
        while True:
            rata_eronare = round(jiwer.wer(propozitii[i],propozitii_eronate[i]),2)

            if rata_eronare >= rata_impusa_eronare or len(t) > round(len(propozitii_eronate[i])/2):
                break
            else:

                propozitii_eronate[i] = jiwer.RemoveEmptyStrings()(propozitii_eronate[i])

                tuplu_eronat = eronare_text(jiwer.SentencesToListOfWords()(propozitii_eronate[i]),t)

                t.append(tuplu_eronat[1])

                propozitii_eronate[i] = listToString(tuplu_eronat[0])
                tuplu_eronat = ()

    return propozitii_eronate

def returnMaxLen(sentences, special_tokens):

    input_ids = torch.tensor(tokenizer.encode(sentences[0], add_special_tokens=special_tokens).ids).unsqueeze(0)
    max_len = input_ids.size()[1]

    for i in range(1, len(sentences)):
        input_ids1= torch.tensor(tokenizer.encode(sentences[i], add_special_tokens = special_tokens).ids).unsqueeze(0)

        if input_ids1.size()[1] >= max_len:
            max_len = input_ids1.size()[1]

    return max_len

def padding(propozitii, special_tokens, length):
    maxx_len = returnMaxLen(propozitii, special_tokens)

    sent_tensor = torch.zeros((length, maxx_len))
    for i in range(length):
        initial_sentance = torch.tensor(tokenizer.encode(propozitii[i], add_special_tokens=special_tokens).ids).unsqueeze(0)
        sentance_padding = torch.stack([torch.cat([i, i.new_zeros(maxx_len - i.size(0))], 0) for i in initial_sentance],1)
        sent_tensor[i] = torch.reshape(sentance_padding, (1, sentance_padding.size()[0]))

    return torch.transpose(sent_tensor,0,1)

def save_checkpoint(state, filename = "checkpointlstmatt.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def correct_sentence(model, sentence, device, max_length):

    sentence_tensor = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True).ids).unsqueeze(1).long().to(device)
    #print(sentence_tensor)

    with torch.no_grad():
        outputs_encoder, hidden, cell = model.encoder(sentence_tensor)

    outputs = [2]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word,outputs_encoder, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == 3:
            break

    translated_sentence = tokenizer.decode(outputs, skip_special_tokens=True)

    # remove start token
    return translated_sentence
def correct_sentence_transformer(model, sentence, device, max_length):

    sentence_tensor = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True).ids).unsqueeze(1).long().to(device)
    #print(sentence_tensor)


    outputs = [2]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == 3:
            break

    translated_sentence = tokenizer.decode(outputs, skip_special_tokens=True)





    return translated_sentence

