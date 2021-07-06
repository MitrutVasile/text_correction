### Generare lista cuvinte des utilizate

import operator

line = open('text_fara_diacritice.txt', 'r', encoding='utf-8').read()


def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

dict1 = word_count(line)
ordonat = dict( sorted(dict1.items(), key=operator.itemgetter(1),reverse=True))

lista = []
for x in ordonat.keys():
    lista.append(x)
l = [x for x in lista if not (x.isdigit()
                                         or x[0] == '-' and x[1:].isdigit())]
print(len(l))
correct_vocab = l
text2 = l[1000:50000]
#Alegerea numărului de propoziții pentru setul de date de antrenare
with open("clean_wiki.txt", 'r', encoding='utf-8') as myfile:
    propozitii = [next(myfile) for x in range(1000000)]

propozitii_eronate  = propozitii.copy()