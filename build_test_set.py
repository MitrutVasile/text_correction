### Script ce construiește setul de date de test

from build_training_dataset import eronare_text2

with open('text_teste.txt', 'r', encoding = 'utf-8') as ff:
    propozitii_corecte = ff.readlines()



str1 = ''
prop_eronate = eronare_text2(propozitii_corecte)




with open('text_teste_gresit70.txt', 'w', encoding = 'utf-8') as ff:
    for i in range(len(prop_eronate)):
        ff.write("%s"%str1.join(prop_eronate[i])+'\n')


