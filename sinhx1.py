import matplotlib.pyplot as plt
import math
import random as rand

# Vstupní parametry
pocet_hodnot = 1000
pocet_neuronu = 100

# Generování vstupních dat
vstup = [rand.uniform(-math.pi, math.pi) for i in range(1000)]
vstup.sort()

# Výpočet požadovaných výstupů
poz_vystup = [math.sinh(x) for x in vstup]

# Generování vah vstupujících do skyté vrstvy
vahy1 = [rand.random() for _ in range(pocet_neuronu)]

# Generování vah vstupujících do výstupního neuronu
vahy2 = [rand.random() for _ in range(pocet_neuronu)]

# Aktivační funkce - hyperbolický tangent
def aktivacni_funkce(x):
    return math.tanh(x)

# Derivace aktivační funkce
def derivace_funkce(x):
    return 1.0 - (aktivacni_funkce(x) ** 2)

#Aktivace neuronů
def aktivace(poradi_vstupu,vstupy, vahy):
    skryta_vrstva = []

    for i in range(len(vahy)):
        vysledek = aktivacni_funkce(vstupy[poradi_vstupu] * vahy[i])
        skryta_vrstva.append(vysledek)
    return skryta_vrstva

# Výstupní vrstva
def vystupni_vrstva(vstupy_skryta, vahy_skryta):
    suma = 0.0
    if len(vstupy_skryta) != len(vahy_skryta):
        return "Vstupy a váhy musí mít stejnou délku"

    for k in range(min(len(vstupy_skryta), len(vahy_skryta))):
        suma += vstupy_skryta[k] * vahy_skryta[k]
    return suma

# Výpčet střední kvadratické chyby
def mse(pozadovane, vypoctene):
    suma = 0.0
    delka_vektoru = min(len(pozadovane), len(vypoctene))
    for i in range(delka_vektoru):
        suma += (pozadovane[i] - vypoctene[i]) ** 2
    return suma / delka_vektoru

# Parametry pro učení
epocha = 0
maximalni_pocet_epoch = 150
soucasne_MSE = 999.0
cilova_hodnota_MSE = 0.01
rychlost_uceni = 0.008

while epocha < maximalni_pocet_epoch and soucasne_MSE > cilova_hodnota_MSE:
    komplet_vysledky = []
    for r in range(len(vstup)):
        vysledna_hodnota = vystupni_vrstva(aktivace(r, vstup, vahy1), vahy2)
        rozdil = poz_vystup[r] - vysledna_hodnota
        komplet_vysledky.append(vysledna_hodnota)

        for i in range(pocet_neuronu):
            delta = rychlost_uceni * rozdil * derivace_funkce(vysledna_hodnota) * vysledna_hodnota
            vahy2[i] += delta

    soucasne_MSE = mse(poz_vystup, komplet_vysledky)
    epocha += 1

    print("Epocha", epocha, "MSE", soucasne_MSE)

# Tvorba grafu pro funkci sinh(x)
plt.plot(vstup, poz_vystup, color='deeppink', marker="o")
plt.plot(vstup, komplet_vysledky, color='blue', marker="o")
plt.title("Graf sinh(x)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


