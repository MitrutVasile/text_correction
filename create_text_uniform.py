def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
line = open('clean_wiki.txt', 'r', encoding='utf-8').read()

line = line.replace('„','')
line = line.replace('”','')
line = line.replace(',','')
line = line.replace(';','')
line = line.replace ('"','')
line = line.replace('“','')
line = line.replace('?','')
line = line.replace('!','')
line = line.replace('.','')

f = open('text_fara_diacritice.txt','w', encoding='utf-8')
f.write(line)

text = []
with open('text_fara_diacritice.txt', 'r', encoding = 'utf-8') as file:
    for line in file:
        for word in line.split():
            text.append(word)



