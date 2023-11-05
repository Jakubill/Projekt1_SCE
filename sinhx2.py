import matplotlib.pyplot as plt
import math
import random as rand

# Aktivační funkce
def aktivacni_funkce(x):
    return math.tanh(x)

# Derivace aktivační funkce
def derivace_akt_funkce(x):
    return 1.0 - (aktivacni_funkce(x) ** 2)

# Vlastnosti neuronové sítě
class NeuronovaSit:
    def __init__(self, skryta_vrstva):
        self.vahy = [rand.random() for _ in range(skryta_vrstva)]
        self.bias = [rand.random() for _ in range(skryta_vrstva)]
        self.vystupni_vahy = [rand.random() for _ in range(skryta_vrstva)]
        self.vystupni_bias = rand.random()

# Dopředný chod
    def dopredne(self, vstupni_vektor):

        vystup_skryte_vrstvy = [aktivacni_funkce(vstupni_vektor * vahy + bias) for vahy, bias in zip(self.vahy, self.bias)]
        vysledek = 0.0
        for i in range(len(vystup_skryte_vrstvy)):
            vysledek += (vystup_skryte_vrstvy[i] * self.vystupni_vahy[i]) + self.vystupni_bias

        return vysledek

# Generování vstupních a testovacích hodnot
pocet_hodnot = 1000
vstup_trenovaci = [rand.uniform(-math.pi, math.pi) for i in range(pocet_hodnot)]
vstup_trenovaci.sort()
vystup_trenovaci = [math.sinh(x) for x in vstup_trenovaci]
#vstup_test = [rand.uniform(-math.pi, math.pi) for k in range(pocet_hodnot)]
#vystup_test = [math.sinh(k) for k in vstup_test]

# Vytvořeí neuronové sítě o n neuronech ve skryté vrstvě
neuronova_sit = NeuronovaSit(50)

def mse(pozadovane, ziskane):
    suma = 0.0
    delka_vektoru = min(len(pozadovane), len(ziskane))
    for t in range(delka_vektoru):
        suma += (pozadovane[t] - ziskane[t]) ** 2
    return suma / delka_vektoru

# Parametry pro učení
epocha = 0
maximalni_pocet_epoch = 500
soucasne_MSE = 999.0
cilova_hodnota_MSE = 0.01
rychlost_uceni = 0.005

while epocha < maximalni_pocet_epoch and soucasne_MSE > cilova_hodnota_MSE:
    vysledek_celkem = []
    for x, y in zip(vstup_trenovaci, vystup_trenovaci):
        vysledek_site = neuronova_sit.dopredne(x)
        vysledek_celkem.append(vysledek_site)
        rozdil = vysledek_site - y
        delta_vstupni = rychlost_uceni * rozdil * derivace_akt_funkce(vysledek_site)
        delta_vystupni = delta_vstupni * x

# Aktualizace vah a biasů
        for i in range(len(neuronova_sit.vahy)):
            neuronova_sit.vahy[i] += delta_vstupni

        for i in range(len(neuronova_sit.bias)):
            neuronova_sit.bias[i] += delta_vstupni

        for i in range(len(neuronova_sit.vystupni_vahy)):
            neuronova_sit.vystupni_vahy[i] += delta_vystupni

        neuronova_sit.vystupni_bias += delta_vystupni

    soucasne_MSE = mse(vystup_trenovaci, vysledek_celkem)
    epocha += 1

    print("Epocha", epocha, "MSE", soucasne_MSE)

# Tvorba grafu pro funkci sinh(x) a její aproximaci
plt.plot(vstup_trenovaci, vystup_trenovaci, color='deeppink', marker="o")
plt.plot(vstup_trenovaci, vysledek_celkem, color='blue', marker="o")
plt.title("Graf sinh(x)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()