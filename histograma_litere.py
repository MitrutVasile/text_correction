### Script ce realizează histograma literelor

import jiwer
import pprint

line = open('text_fara_diacritice.txt', 'r', encoding='utf-8').read()

from collections import Counter
import matplotlib.pyplot as plt



line = line.lower()

counts=Counter(line) # Counter({'l': 2, 'H'1, 'e': 1, 'o': 1})
dictionar_litere_sortate = counts.keys()
keys_to_remove = ["1", "2", "3", "4", "5", "6", "7","8", "9","0","-", " ", '\n']
for keys in keys_to_remove:

    counts.pop(keys)
plt.bar(*zip(*counts.items()))
plt.show()
pprint.pprint((counts))
lista = list({k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse= True)}.keys())
lista = jiwer.RemoveSpecificWords(["1", "2", "3", "4", "5", "6", "7","8", "9","0","„","”","-",])(lista)
lista = jiwer.RemoveEmptyStrings()(lista)


dict_litere = lista [1:15]

