from random import randrange
from csv import reader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Razdeli dataset na št. foldov, ki jih določiš
def razdelitevNaFolde(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def izracunTocnosti(pravilni, predvidevani):
    pravilnoKlasificirani = 0
    for i in range(len(pravilni)):
        if pravilni[i] == predvidevani[i]:
            pravilnoKlasificirani += 1
    return pravilnoKlasificirani / float(len(pravilni)) * 100.0

def izracunSenzitivnosti(pravilni, predvidevani):
    TP = 0
    FN = 0
    for i in range(len(pravilni)):
        if pravilni[i] == 1 and predvidevani[i] == 1:
            TP += 1
        elif pravilni[i] == 1 and predvidevani[i] == 0:
            FN += 1
    return (TP / (TP + FN)) * 100.0

def izracunSpecificnosti(pravilni, predvidevani):
    TN = 0
    FP = 0
    for i in range(len(pravilni)):
        if pravilni[i] == 0 and predvidevani[i] == 0:
            TN += 1
        elif pravilni[i] == 0 and predvidevani[i] == 1:
            FP += 1
    return (TN / (TN + FP)) * 100.0

def izracunMatrikeZmede(pravilni, predvidevani):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    matrikaZmede = list()
    for i in range(len(pravilni)):
        if pravilni[i] == 0 and predvidevani[i] == 0:
            TN += 1
        elif pravilni[i] == 1 and predvidevani[i] == 1:
            TP += 1
        elif pravilni[i] == 0 and predvidevani[i] == 1:
            FP += 1
        elif pravilni[i] == 1 and predvidevani[i] == 0:
            FN += 1
    matrikaZmede.append(TP)
    matrikaZmede.append(FN)
    matrikaZmede.append(FP)
    matrikaZmede.append(TN)
    return matrikaZmede

# Vrne rezultate (acc, sen, spec, matrike zmede)
def oceniAlgoritem(foldi, stSosedov):
    metrike = list()
    senzitivnosti = list()
    tocnosti = list()
    specificnosti = list()
    matrikaZmede = list()

    for fold in foldi:
        ucnaMnozica = list(foldi)
        ucnaMnozica.remove(fold)
        ucnaMnozica = sum(ucnaMnozica, []) # združi folde v en array
        testnaMnozica = list()
        # napolni testno množico
        for row in fold:
            testnaMnozica.append(row)
        # napove razrede na podlagi algoritma <K najbližjih sosedov>
        napovedani = izvediAlgoritem(ucnaMnozica, testnaMnozica, stSosedov)
        # pravilno določeni razredi
        pravilni = list()
        for row in fold:
            pravilni.append(row[-1])

        # izračuni metrik
        tocnosti.append(izracunTocnosti(pravilni, napovedani))
        senzitivnosti.append(izracunSenzitivnosti(pravilni, napovedani))
        specificnosti.append(izracunSpecificnosti(pravilni, napovedani))
        matrikaZmede.append(izracunMatrikeZmede(pravilni, napovedani))
        izracunAUC(pravilni, napovedani) # izriše graf ROC


    metrike.append(tocnosti)
    metrike.append(senzitivnosti)
    metrike.append(specificnosti)
    metrike.append(matrikaZmede)
    return metrike

# Formula -> vsota od 1 do n |Xi - Yi|
def izracunRazdaljeManhattan(testnaVrstica, ucnaVrstica):
    razdalja = 0
    for i in range(len(testnaVrstica)-1):
        razdalja += abs(testnaVrstica[i] - ucnaVrstica[i])
    return razdalja

# Doloci najbolj pogost razred izmed k podobnih sosedov
def pridobiNapovedanRazred(ucnaMnozica, testnaVrstica, stSosedov):
    SosedInRazdalja = list()
    for ucnaVrstica in ucnaMnozica:
        razdalja = izracunRazdaljeManhattan(testnaVrstica, ucnaVrstica)
        SosedInRazdalja.append((ucnaVrstica, razdalja))
    SosedInRazdalja.sort(key=lambda x: x[1]) # sortira razdalje od min do max
    sosedi = list()
    for i in range(stSosedov):
        sosedi.append(SosedInRazdalja[i][0])

    razrediSosedov = list()
    for row in sosedi:
        razrediSosedov.append(row[-1])

    napovedanRazred = max(set(razrediSosedov), key=razrediSosedov.count) # določi najbolj pogost razred med sosedi

    return napovedanRazred

# K najbližjih sosedov
def izvediAlgoritem(ucnaMnozica, testnaMnozica, stSosedov):
    napovedani = list()
    for testnaVrstica in testnaMnozica:
        napovedanRazred = pridobiNapovedanRazred(ucnaMnozica, testnaVrstica, stSosedov)
        napovedani.append(napovedanRazred)

    return napovedani

def izracunAUC(pravilni, napovedani):
    fpr, tpr, _ = roc_curve(pravilni, napovedani)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

def boxPlot(data):
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(data)
    # show plot
    plt.show()

#########################################################

ime = '../spambase.csv'
dataset = list()

# Prebere csv datoteko
with open(ime, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)

# Pretvori vsaki stolpec iz string v float, razen zadnjega
for vrstica in dataset:
    for stolpec in range(len(dataset[0])-1):
        vrstica[stolpec] = float(vrstica[stolpec].strip())

# Pretvori zadnji stolpec iz string v int (razredi)
for vrstica in dataset:
    if vrstica[len(dataset[0])-1] == "1":
        vrstica[len(dataset[0])-1] = 1
    elif vrstica[len(dataset[0])-1] == "0":
        vrstica[len(dataset[0])-1] = 0

stFoldov = 5
stSosedov = 3
foldi = razdelitevNaFolde(dataset, stFoldov)
# rezultati: acc, senz, spec, matrike zmede
rezultati = oceniAlgoritem(foldi, stSosedov)
print('Matrike zmede: %s' % rezultati[3])
print('----------------------------------')
print('Acc: %s' % rezultati[0])
print('Povprečni acc: %.3f%%' % (sum(rezultati[0])/float(len(rezultati[0]))))
print('Senz: %s' % rezultati[1])
print('Povprečna senz: %.3f%%' % (sum(rezultati[1])/float(len(rezultati[1]))))
print('Spec: %s' % rezultati[2])
print('Povprečna spec: %.3f%%' % (sum(rezultati[2])/float(len(rezultati[2]))))

print('----------------------------------')

priklici1 = list()
priklici2 = list()
preciznosti1 = list()
preciznosti2 = list()
f_mera1 = list()
f_mera2 = list()

# izračun recall, precision, f-mera iz matrik zmede
for matrika in rezultati[3]:
    priklic1 = matrika[0] / (matrika[0] + matrika[1])
    priklic2 = matrika[3] / (matrika[2] + matrika[3])
    prec1 = matrika[0] / (matrika[0] + matrika[2])
    prec2 = matrika[3] / (matrika[3] + matrika[1])
    priklici1.append(priklic1 * 100)
    priklici2.append(priklic2 * 100)
    preciznosti1.append(prec1 * 100)
    preciznosti2.append(prec2 * 100)
    f_mera1.append(2 * ((prec1 * priklic1) / (prec1 + priklic1)))
    f_mera2.append(2 * ((prec2 * priklic2) / (prec2 + priklic2)))

# združi lista skupaj
recall = [y for x in [priklici1, priklici2] for y in x]
precision = [y for x in [preciznosti1, preciznosti2] for y in x]
fMera = [y for x in [f_mera1, f_mera2] for y in x]

print(recall)
print('Recalls (1. razred -> pozitivni): %s' % priklici1)
print('Recalls (2. razred -> negativni): %s' % priklici2)
print('Povprečni recall: %.3f%%' % (sum(recall)/float(len(recall))))
print('Precision (1. razred -> pozitivni): %s' % preciznosti1)
print('Precision (2. razred -> negativni): %s' % preciznosti2)
print('Povprečni precision: %.3f%%' % (sum(precision)/float(len(precision))))
print('F-mere (1. razred -> pozitivni): %s' % f_mera1)
print('F-mere (2. razred -> negativni): %s' % f_mera2)
print('Povprečna F-mera: %.3f%%' % (sum(fMera)/float(len(fMera))))

# združi vse metrike
data = [rezultati[0], rezultati[1], rezultati[2], recall, precision, fMera]
# prikaži boxplot graf
boxPlot(data)

