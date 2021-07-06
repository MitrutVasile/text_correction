### Script ce construiește setul de date

from sortareFrecventaCuvinte import propozitii
from build_training_dataset import eronare_text2
from itertools import repeat
import random
propozitii_corecte = [x for item in propozitii for x in repeat(item, 8)]
str1 = ""
propozitii_eronate = []

for i in range(len(propozitii_corecte)):
    if i % 8 == 0:
        print('s-a eronat propozitia', i//8 + 1, 'din', len(propozitii))
        prop_corecte = propozitii_corecte [i:i+8]
        prop_eronate = eronare_text2(prop_corecte)
        propozitii_eronate = propozitii_eronate + prop_eronate

c = list(zip(propozitii_eronate, propozitii_corecte))
random.shuffle(c)
propozitii_eronate, propozitii_corecte = zip(*c)

with open('target.txt', 'w', encoding = 'utf-8') as ff:
    for i in range(len(propozitii_corecte)):
        ff.write("%s"%str1.join(propozitii_corecte[i]))

str2 = ""
with open('soruce.txt', 'w', encoding='utf-8') as ff:
    for i in range(len(propozitii_eronate)):
        ff.write("%s" % str1.join(propozitii_eronate[i])+ "\n")
