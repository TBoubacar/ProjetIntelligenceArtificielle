import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

#   RECUPERATION DES JEUX DE DONNÉES
# nomFichierCSV = "iris2Classes.csv"
# nomFichierCSV = "heart-statlog.csv"
nomFichierCSV = "diabetes.csv"

infoData = pd.read_csv(nomFichierCSV, sep=",")
classTrain = infoData['class']
valueTrain = infoData.drop('class', axis=1)

#   CREATION DES FONCTIONS DE TRAITEMENTS DE DONNÉES
def dessinneGraphe(nomClass1, nomClass2, nomAttribut1, nomAttribut2, titre):
    for e, c in [(nomClass1, 'g'), (nomClass2, 'r')]:
        plt.scatter(valueTrain.loc[:, nomAttribut1][classTrain == e], valueTrain.loc[:, nomAttribut2][classTrain == e], alpha=0.5, color=c, label=e)
    plt.xlabel(nomAttribut1)
    plt.ylabel(nomAttribut2)
    plt.legend()
    plt.title(titre)
    plt.show()

def affichageGraphiqueIris(nomDuFichier):
    '''
    :param nomDuFichier:    CORRESPOND AU NOM DU FICHIER CSV CONTENANT LES JEUX DE DONNÉES A ETUDIER
    :return:                AFFICHE GRACE À LA BIBLIOTHEQUE MATPLOTLIB UNE REPRESENTATION DES DIFFÉRENTES CLASSES DE NOTRE JEU DE DONNÉE DANS UN GRAPHISME
    '''
    print("=========Affichage graphique du jeu de données (", nomDuFichier, ") :")
    print("L'attribut le plus important est : petallength (attribut n°3)\n")
    dessinneGraphe('Iris-versicolor', 'Iris-virginica', "petallength", "petalwidth", "Les K plus proches voisins pour le jeu de données ({})".format(nomDuFichier))

def affichageGraphiqueHeartStatlog(nomDuFichier):
    '''
    :param nomDuFichier:    CORRESPOND AU NOM DU FICHIER CSV CONTENANT LES JEUX DE DONNÉES A ETUDIER
    :return:                AFFICHE GRACE À LA BIBLIOTHEQUE MATPLOTLIB UNE REPRESENTATION DES DIFFÉRENTES CLASSES DE NOTRE JEU DE DONNÉE DANS UN GRAPHISME
    '''
    print("=========Affichage graphique du jeu de données (", nomDuFichier, ") : ...")
    print("L'attribut le plus important est : maximum_heart_rate_achieved (attribut n°8)\n")
    dessinneGraphe('absent', 'present', "age", "maximum_heart_rate_achieved", "Les K plus proches voisins pour le jeu de données ({})".format(nomDuFichier))

def affichageGraphiqueDiabete(nomDuFichier):
    '''
    :param nomDuFichier:    CORRESPOND AU NOM DU FICHIER CSV CONTENANT LES JEUX DE DONNÉES A ETUDIER
    :return:                AFFICHE GRACE À LA BIBLIOTHEQUE MATPLOTLIB UNE REPRESENTATION DES DIFFÉRENTES CLASSES DE NOTRE JEU DE DONNÉE DANS UN GRAPHISME
    '''
    print("=========Affichage graphique du jeu de données (", nomDuFichier, ") : ...")
    print("L'attribut le plus important est : age (attribut n°8) et pres (attribut n°3)\n")
    dessinneGraphe('tested_negative', 'tested_positive', "age", "pres", "Les K plus proches voisins pour le jeu de données ({})".format(nomDuFichier))

def affichageDesDonnees(nomDuFichier):
    '''
    :param nomDuFichier:    CORRESPOND AU NOM DU FICHIER CSV CONTENANT LES JEUX DE DONNÉES A ETUDIER
    :return:                AFFICHE L'ENSEMBLE DES INFORMATIONS SUR LE JEU DE DONNÉES FOURNI
    '''
    if nomDuFichier == "iris2Classes.csv":
        affichageGraphiqueIris(nomFichierCSV)
    if nomDuFichier == "heart-statlog.csv":
        affichageGraphiqueHeartStatlog(nomDuFichier)
    if nomDuFichier == "diabetes.csv":
        affichageGraphiqueDiabete(nomDuFichier)
    print("=========Affichage du jeux de données (", nomDuFichier, ") :")
    print(infoData, "\n")
    print("---------Affichage des Classes à rechercher :")
    print(classTrain, "\n")
    print("---------Affichage des valeurs à étudier :")
    print(valueTrain)
    print("=========Fin de l'affichage du jeu de données\n")

def initialisationDuVecteurPoidsDesAttributs():
    '''
    :return:    FONCTION PERMETTANT D'INITIALISER LE VECTEUR DE POIDS DES ATTRIBUTS DE NOTRE JEU DE DONNÉES A 0 (W[i] = 0)
    '''
    print("=========Initialisation du vecteur poids W[i] à 0, pour i compris entre [0, d] avec d le nombre d'attributs du jeu de données choisi.")
    #   DECLARATION ET INITIALISATION DU VECTEUR DE POIDS POUR L'ENSEMBLE DES ATTRIBUTS DE NOTRE JEU DE DONNÉES
    W = np.zeros(len(infoData.columns)-1)
    print(W, '\n')
    return W

def normaliserLesDonneesDesAttributs():
    '''
    :return:    RETOURNE UN TABLEAU CONTENANT DES ENSEMBLES DE VECTEUR NORMALISÉ DES DONNÉES DE NOS ATTRIBUTS
    '''
    print("Normalisation des valeurs des attributs fait !\n")
    tab = []
    for sample in valueTrain.values:
        tab.append(normalize(sample[:, np.newaxis], axis=0).ravel())
    return tab

def distanceEuclidienne(x, y):
    '''
    :param x:   VECTEUR ARRAY DE TAILLE Xn
    :param y:   VECTEUR ARRAY DE TAILLE Yn  (SACHANT QUE LES PARAMETRES DOIVENT FORCEMENT RESPECTER LA CONDITION SUIVANTE : Xn = Yn)
    :return:    CETTE FONCTION RETOURNE LA DISTANCE EUCLIDIENNE ENTRE LES DEUX VECTEURS PASSÉS EN PARAMÈTRE
    '''
    assert len(x) == len(y), "Calcul de distance impossible entre deux vecteurs de taille différente."
    return math.sqrt(sum((np.array(x)-np.array(y))**2))

def KNNOfSameClasse(dataValueTrain, x, k, typeClass):
    '''
    :param dataValueTrain:  VECTEUR CONTENANT LES VALEURS A ETUDIER (ENSEMBLE DE VECTEUR DE DONNÉES DES ATTRIBUTS)
    :param x:               VECTEUR AVEC LAQUELLE ON DOIT APPLIQUER LE CALCUL DE DISTANCE
    :param k:               LE NOMBRE DE VOISINS A RETOURNER
    :param typeClass:       LA CLASSE DU VECTEUR SUR LAQUELLE ON DOIT APPLIQUER LE CALCUL DE DISTANCE (x)
    :return:                CETTE FONCTION RETOURNE L'INDICE DES K PLUS PROCHES VOISINS DE MEME CLASSE QUE LE VECTEUR x DANS L'ENSEMBLE DES VALEURS DE NOTRE JEU DE DONNÉES (dataValueTrain)
    '''
    distancesVoisins = []
    for index, sample in enumerate(dataValueTrain):
        if not np.array_equal(x, sample):
            distance = distanceEuclidienne(sample, x)
            distancesVoisins.append((distance, index))
    distancesVoisins = sorted(distancesVoisins)
    # print("Affichage des", k, "plus proches voisins de", x, "de même classe que '", typeClass, "' :")
    distancesVoisinsTmp = []
    indiceKNN = []
    for distance, index in distancesVoisins:
        if classTrain[index] == typeClass:
            distancesVoisinsTmp.append((distance, index))
            indiceKNN.append(index)
    # print(distancesVoisinsTmp, "\n")
    return indiceKNN[: k]

def KNNOfDiffClasse(dataValueTrain, x, k, typeClass):
    '''
    :param dataValueTrain:  VECTEUR CONTENANT LES VALEURS A ETUDIER (ENSEMBLE DE VECTEUR DE DONNÉES DES ATTRIBUTS)
    :param x:               VECTEUR AVEC LAQUELLE ON DOIT APPLIQUER LE CALCUL DE DISTANCE
    :param k:               LE NOMBRE DE VOISINS A RETOURNER
    :param typeClass:       LA CLASSE DU VECTEUR SUR LAQUELLE ON DOIT APPLIQUER LE CALCUL DE DISTANCE (x)
    :return:                CETTE FONCTION RETOURNE L'INDICE DES K PLUS PROCHES VOISINS DE CLASSE DIFFERENTE DE CELLE DU VECTEUR x DANS L'ENSEMBLE DES VALEURS DE NOTRE JEU DE DONNÉES (dataValueTrain)
    '''
    distancesVoisins = []
    for index, sample in enumerate(dataValueTrain):
        if not np.array_equal(x, sample):
            distance = distanceEuclidienne(sample, x)
            distancesVoisins.append((distance, index))
    distancesVoisins = sorted(distancesVoisins)
    # print("Affichage des", k, "plus proches voisins de", x, "de classe différente de '", typeClass, "' :")
    distancesVoisinsTmp = []
    indiceKNN = []
    for distance, index in distancesVoisins:
        if classTrain[index] != typeClass:
            distancesVoisinsTmp.append((distance, index))
            indiceKNN.append(index)
    # print(distancesVoisinsTmp, "\n")
    return indiceKNN[: k]

def traitementDuPseudoCodeRelief(W):
    m = random.randint(20, 30)
    k = 1   #   LES K PLUS PROCHES VOISINS
    print("=========Traitement de l'algorithme Relief pour m =", m, "k =", k)
    tabAttNormaliser = normaliserLesDonneesDesAttributs()
    for i in range(0, m):
        xIndice = random.randint(0, len(tabAttNormaliser) - 1)
        X = tabAttNormaliser[xIndice]
        typeOfX = classTrain.values[xIndice]

        indiceOfNearestHit = KNNOfSameClasse(tabAttNormaliser, X, k, typeOfX)[0]
        nearestHit = tabAttNormaliser[indiceOfNearestHit]

        indiceOfNearestMiss = KNNOfDiffClasse(tabAttNormaliser, X, k, typeOfX)[0]
        nearestMiss = tabAttNormaliser[indiceOfNearestMiss]

        # print("\tElement tiré aléatoirement :", xIndice, X, typeOfX, "\n\tLe plus proche voisins de même classe :", indiceOfNearestHit, nearestHit, classTrain.values[indiceOfNearestHit], "\n\tLe plus proche voisins de classe différente :", indiceOfNearestMiss, nearestMiss, classTrain.values[indiceOfNearestMiss], "\n")

        for j in range(0, len(W)):
            W[j] = W[j] - (1/m) * math.pow(X[j] - nearestHit[j], 2) + (1/m) * math.pow(X[j] - nearestMiss[j], 2)
    print("_____________________________________________________________________________________________________________")
    print("|\tPlus le poids des attributs sont forts, plus l'attribut est important pour la classification !\n|\tLes poids W[i] des", len(W), "attributs sont :", W)
    print("_____________________________________________________________________________________________________________\n")

#==========================     PARTIE BONUS (AMELIORATION DE L'ALGORITHME)     ==========================

def determineBarycentre(vecteurKNN, tabAttribut):
    '''
    :param vecteurKNN:      CONTIENT L'INDICE DES K PLUS PROCHES VOISINS
    :param tabAttribut:     CONTIENT L'ENSEMBLE DES VALEURS DES ATTRIBUTS
    :return:                DETERMINE LE BARYCENTRE DES VECTEURS (LE VECTEUR BARYCENTRE)
    '''
    barycentre = []
    for i in range(0, valueTrain.shape[1]):
        coordonnee = 0
        nbValeur = 0
        for j in vecteurKNN:
            nbValeur += 1
            coordonnee += tabAttribut[j][i]
        coordonnee /= nbValeur
        barycentre.append(coordonnee)
    return barycentre

def traitementDuPseudoCodeReliefAvecBarycentre(W, k):
    m = len(valueTrain.values)
    print("=========Traitement de l'algorithme Relief (avec Amélioration et Intégration du Barycentre) pour m =", m, "k =", k)
    tabAttNormaliser = normaliserLesDonneesDesAttributs()
    for i in range(0, m):
        xIndice = random.randint(0, len(tabAttNormaliser) - 1)
        X = tabAttNormaliser[xIndice]
        typeOfX = classTrain.values[xIndice]

        indiceOfNearestHit = KNNOfSameClasse(tabAttNormaliser, X, k, typeOfX)
        nearestHit = determineBarycentre(indiceOfNearestHit, tabAttNormaliser)

        indiceOfNearestMiss = KNNOfDiffClasse(tabAttNormaliser, X, k, typeOfX)
        nearestMiss = determineBarycentre(indiceOfNearestMiss, tabAttNormaliser)

        # print("\tElement tiré aléatoirement :", xIndice, X, typeOfX, "\n\tLe(s) plus proche(s) voisin(s) de même classe :", indiceOfNearestHit, nearestHit, classTrain.values[indiceOfNearestHit], "\n\tLe(s) plus proche(s) voisin(s) de classe différente :", indiceOfNearestMiss, nearestMiss, classTrain.values[indiceOfNearestMiss], "\n")

        for j in range(0, len(W)):
            W[j] = W[j] - (1/m) * math.pow(X[j] - nearestHit[j], 2) + (1/m) * math.pow(X[j] - nearestMiss[j], 2)
    print("_____________________________________________________________________________________________________________")
    print("|\tPlus le poids des attributs sont forts, plus l'attribut est important pour la classification !\n|\tLes poids W[i] des", len(W), "attributs sont :", W)
    print("_____________________________________________________________________________________________________________\n")

#   DEBUT DU PROGRAMME PRINCIPAL
if __name__ == '__main__':
    affichageDesDonnees(nomFichierCSV)
    W = initialisationDuVecteurPoidsDesAttributs()
    traitementDuPseudoCodeRelief(W)
    print("\t_______ALGORITHME AMELIORER AVEC INTEGRATION DU CALCUL DE BARYCENTRE_______\n")
    traitementDuPseudoCodeReliefAvecBarycentre(W, 5)