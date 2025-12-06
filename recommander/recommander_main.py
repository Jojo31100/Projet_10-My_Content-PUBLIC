import pandas
import pickle
import os
import numpy
import psutil
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import implicit
import time
from azure.storage.blob import BlobServiceClient
import io
import logging


#Configuration du logger pour des messages clairs
logger = logging.getLogger("recommander_main")
logger.setLevel(logging.INFO)
#Désactivation du log de base d'Azure, pour réduire le "bruit de fond" (Azure étant encore plus bavard que moi... o_O )
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


#Variables globales (stockage en mémoire de la fonction Azure)
clicsUtilisateur = None
articlesDataframe = None
article_ids = None
tableauEmbeddings = None
article_id_to_index = None
articleParCategorie = None
modeleALS = None
matriceArticles_x_Utilisateurs = None
user_id_to_index = None
index_to_article_id = None


#Connexion Azure Blob Storage (initialisation unique)
chaineDeConnexion = os.getenv("chaineDeConnexion")
if(not chaineDeConnexion):
    raise ValueError("La variable d'environnement \"chaineDeConnexion\" n'est pas définie !")
nomDuConteneur = "data"
clientBlobStorageService = BlobServiceClient.from_connection_string(chaineDeConnexion)
clientConteneur = clientBlobStorageService.get_container_client(nomDuConteneur)


#######################################
# Chargement des données au démarrage #
#######################################
def _chargerDepuisBlobStorage(nomDuBlob):
    #Télécharge le contenu d'un blob en mémoire
    try:
        blob_client = clientConteneur.get_blob_client(nomDuBlob)
        fichierBrut = blob_client.download_blob().readall()
        return fichierBrut
    except Exception as excpt:
        logger.error(f"Erreur de chargement du blob {nomDuBlob}: {excpt}")
        return None

def _chargementDesDonnees():
    global clicsUtilisateur, articlesDataframe, tableauEmbeddings, article_id_to_index, articleParCategorie, article_ids
    #Si les données sont déjà disponibles en mémoire, inutile de tout recharger
    if(clicsUtilisateur is not None):
        logger.info("[Initialisation] Données déjà en mémoire, aucun rechargement nécessaire.")
        return
    logger.info("[Initialisation] Chargement complet des données depuis Blob Storage...")
    #Utilisation d'un dictionnaire temporaire pour s'assurer que toutes les données sont chargées avant de les assigner
    clicsUtilisateurTemp = {}
    article_idsTemp = None
    articlesDataframeTemp = None
    tableauEmbeddingsTemp = None
    article_id_to_indexTemp = None
    articleParCategorieTemp = None
    #CLICS
    logger.info("  -> Chargement des clics utilisateurs...")
    nbFichiers = 385
    fichiers_charges = 0
    for numFichier in range(nbFichiers):
        nomDuBlob = f"clicks/clicks_hour_{numFichier:03d}.csv"
        fichierBrut = _chargerDepuisBlobStorage(nomDuBlob)
        if(fichierBrut is None):
            continue #Passe au fichier suivant si l'erreur est non critique (fichier manquant)
        try:
            clicEnCours = pandas.read_csv(io.BytesIO(fichierBrut))
            for _, row in clicEnCours.iterrows():
                user_id = row["user_id"]
                article_id = row["click_article_id"]
                clicsUtilisateurTemp.setdefault(user_id, {}).setdefault(article_id, 0)
                clicsUtilisateurTemp[user_id][article_id] += 1
            fichiers_charges += 1
            if(fichiers_charges % 50 == 0):
                logger.info(f"    --> {fichiers_charges}/{nbFichiers} fichiers de clics traités...")
        except Exception as excpt:
            logger.warning(f"Avertissement : Impossible de lire le fichier CSV {nomDuBlob}: {excpt}")
    if(not clicsUtilisateurTemp):
        raise RuntimeError("Aucun clic n'a pu être chargé ! Vérifiez la connexion et le chemin du blobStorage...")
    logger.info(f"  > {len(clicsUtilisateurTemp)} utilisateurs chargés : OK")
    #ARTICLES
    logger.info("  -> Chargement des articles...")
    fichierBrut = _chargerDepuisBlobStorage("articles_metadata.csv")
    if(fichierBrut is None):
        raise RuntimeError("Échec critique : Impossible de charger \"articles_metadata.csv\".")
    articlesDataframeTemp = pandas.read_csv(io.BytesIO(fichierBrut))
    article_idsTemp = articlesDataframeTemp["article_id"].values
    articleParCategorieTemp = dict(zip(articlesDataframeTemp["article_id"], articlesDataframeTemp["category_id"]))
    logger.info(f"  > {len(article_idsTemp)} articles chargés : OK")
    #EMBEDDINGS
    logger.info("  -> Chargement des embeddings...")
    fichierBrut = _chargerDepuisBlobStorage("articles_embeddings.pickle")
    if(fichierBrut is None):
        raise RuntimeError("Échec critique : Impossible de charger \"articles_embeddings.pickle\".")
    tableauEmbeddingsTemp = pickle.loads(fichierBrut)
    #On vérifie que le nombre d'embeddings correspond bien au nombre d'articles
    if(tableauEmbeddingsTemp.shape[0] != len(article_idsTemp)):
        raise RuntimeError("Problème détecté : Le nombre d'embeddings ne correspond pas au nombre d'articles !")
    article_id_to_indexTemp = {article_id: index for index, article_id in enumerate(article_idsTemp)}
    logger.info(f"  > {tableauEmbeddingsTemp.shape[0]} embeddings chargés : OK")
    #Assignation globale (seulement si tout les chargements ont réussi...)
    clicsUtilisateur = clicsUtilisateurTemp
    articlesDataframe = articlesDataframeTemp
    article_ids = article_idsTemp
    tableauEmbeddings = tableauEmbeddingsTemp
    article_id_to_index = article_id_to_indexTemp
    articleParCategorie = articleParCategorieTemp
    logger.info("[Initialisation] Chargement terminé avec succès.")


###########################
# Content-Based Filtering #
###########################
def _creationProfilUtilisateur(user_id, clicsUtilisateur):
    if(user_id not in clicsUtilisateur):
        return None
    articlesCliques = clicsUtilisateur[user_id]
    #Récupération des embeddings et des poids
    listeEmbeddings = []
    listePoids = []
    for article_id, nbClics in articlesCliques.items():
        if(article_id in article_id_to_index):
            index = article_id_to_index[article_id]
            listeEmbeddings.append(tableauEmbeddings[index])
            listePoids.append(nbClics)
    if(len(listeEmbeddings)==0):
        return None
    #Conversion en Array NumPy
    matriceEmbeddings = numpy.array(listeEmbeddings)
    poids = numpy.array(listePoids)
    #Normalisation des poids (somme = 1)
    poidsNormalises = poids / poids.sum()
    #Calcul de la moyenne pondérée
    profilUtilisateur = numpy.average(matriceEmbeddings, axis=0, weights=poidsNormalises)
    return profilUtilisateur

def _calculSimilarities(profilUtilisateur):
    #Réorganisation du profil utilisateur pour SKLearn
    profilUtilisateurReorg = profilUtilisateur.reshape(1,-1)
    #Calcul de la similarité Cosinus
    similarites = cosine_similarity(profilUtilisateurReorg, tableauEmbeddings)[0]
    return similarites

def _recommandationsContent(user_id, clicsUtilisateur, nbReco=5):
    #Vérification de l'existence de l'utilisateur
    if(user_id not in clicsUtilisateur):
        logger.warning(f"Utilisateur {user_id} non connu pour Content-Based !")
        return []
    #Création du profil utilisateur
    profilUtilisateur = _creationProfilUtilisateur(user_id, clicsUtilisateur)
    if(profilUtilisateur is None):
        logger.warning(f"Problème de création du profil utilisateur : {user_id}")
        return []
    #Calcul des similarités
    similarites = _calculSimilarities(profilUtilisateur)
    #Récupération des articles déjà lus
    articlesDejaLus = set(clicsUtilisateur[user_id].keys())
    #Création d'une liste (article_id, similarité)
    recommandations = []
    for index, scoreSimilarite in enumerate(similarites):
        article_id = article_ids[index]
        if(article_id not in articlesDejaLus):
            recommandations.append((article_id, scoreSimilarite))
    #Tri par score décroissant
    recommandations.sort(key=lambda x: x[1], reverse=True)
    #Retour du top "nbReco"
    top_nbReco = [{"article_id": article_id, "score": float(score)} for article_id, score in recommandations[:nbReco]]
    return top_nbReco

#################################
# Collaborative-Based Filtering #
#################################
def _entrainementCollaboratif(clicsUtilisateur):
    global modeleALS, matriceArticles_x_Utilisateurs, user_id_to_index, index_to_article_id
    #Si le modèle existe déjà, c'est qu'il a déjà été entraîné ... Donc on peut zapper !
    if(modeleALS is not None):
        logger.info("[Collaborative-Based Filtering] Modèle déjà en mémoire, aucun réentraînement nécessaire...")
        return modeleALS, matriceArticles_x_Utilisateurs, user_id_to_index, index_to_article_id
    else:
        logger.info("[Collaborative-Based Filtering] Entraînement du modèle ALS...")
        #Création des mappings
        user_id_to_index = {user_id: index for index, user_id in enumerate(clicsUtilisateur.keys())}
        articlesCliques = set(article_id for articles in clicsUtilisateur.values() for article_id in articles.keys())
        article_id_to_index_collab = {article_id: index for index, article_id in enumerate(sorted(articlesCliques))}
        index_to_article_id = {index: article_id for article_id, index in article_id_to_index_collab.items()}
        #Construction des données pour la matrice sparse
        lignes = []
        colonnes = []
        scores = []
        for user_id, articles in clicsUtilisateur.items():
            user_index = user_id_to_index[user_id]
            for article_id, nbClics in articles.items():
                if(article_id in article_id_to_index_collab):
                    article_index = article_id_to_index_collab[article_id]
                    lignes.append(user_index)
                    colonnes.append(article_index)
                    scores.append(nbClics)
        #Création de la matrice au format CSR
        matriceArticles_x_Utilisateurs = csr_matrix((scores, (lignes, colonnes)), shape=(len(user_id_to_index), len(article_id_to_index_collab)))
        #Configuration du modèle
        modeleALS = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20, random_state=23011977)
        #Entraînement
        modeleALS.fit(matriceArticles_x_Utilisateurs)
        logger.info("[Collaborative-Based Filtering] Modèle entraîné et stocké en mémoire !")
    return modeleALS, matriceArticles_x_Utilisateurs, user_id_to_index, index_to_article_id

def _recommandationsCollaborative(user_id, modeleALS, matrice, user_id_to_index, index_to_article_id, nbReco=5):
    if(user_id not in user_id_to_index):
        logger.warning(f"Utilisateur {user_id} non présent dans le modèle Collaborative !")
        return []
    user_index = user_id_to_index[user_id]
    indicesDesRecommandations, scoresDesRecommandations = modeleALS.recommend(user_index, matrice[user_index], N=nbReco, filter_already_liked_items=True)
    recommandations = []
    for article_index, score in zip(indicesDesRecommandations, scoresDesRecommandations):
        article_id = index_to_article_id[article_index]
        recommandations.append({"article_id": article_id, "score": float(score)})
    return recommandations


##############################
# Popularity-Based Filtering #
##############################
def _recommandationsPopularity(user_id, clicsUtilisateur, nbReco=5):
    articlesGlobal = {}
    for data in clicsUtilisateur.values():
        for articleID, num in data.items():
            articlesGlobal[articleID] = articlesGlobal.get(articleID, 0) + num
    articlesDejaLus = set(clicsUtilisateur.get(user_id, {}).keys())
    recommandations = [(articleID, score) for articleID, score in articlesGlobal.items() if articleID not in articlesDejaLus]
    recommandations.sort(key=lambda x: x[1], reverse=True)
    top_nbReco = [{"article_id": int(articleID), "score": float(score)} for articleID, score in recommandations[:nbReco]]
    return top_nbReco


#############################
# Fonctions supplémentaires #
#############################
def _getStatsUtilisateur(user_id, clicsUtilisateur):
    articles = clicsUtilisateur.get(user_id,{})
    nbClics = sum(articles.values())
    nbArticles = len(articles)
    categories = {articleParCategorie.get(articleID) for articleID in articles if articleID in articleParCategorie}
    nbCategories = len(categories)
    return {"clics": nbClics, "articles": nbArticles, "categories": nbCategories}

def _choixModele(stats):
    if stats["clics"] < 5:
        logger.info("Popularity-based Filtering")
        return "popularity"
    elif stats["clics"] >= 5 and stats["categories"] < 3:
        logger.info("Content-Based Filtering")
        return "content"
    else: #équivalent à : stats["clics"] >= 5 and stats["categories"] >= 3
        logger.info("Collaborative Filtering")
        return "collaborative"

def _convert(x):
    if(isinstance(x, (numpy.integer, numpy.int32, numpy.int64))):
        return int(x)
    if(isinstance(x, (numpy.floating, numpy.float32, numpy.float64))):
        return float(x)
    return x


##############################################
# Fonction principale appelée par Azure Func #
##############################################
def get_recommendations(user_id: int, nbReco: int = 5):
    global clicsUtilisateur, articlesDataframe, tableauEmbeddings, article_id_to_index, articleParCategorie, modeleALS, matriceArticles_x_Utilisateurs, user_id_to_index, index_to_article_id
    try:
        timerStart = time.time()
        #Chargement des données (uniquement si pas déjà en mémoire)
        _chargementDesDonnees()
        #Vérifications critiques après chargement
        if(clicsUtilisateur is None):
            raise RuntimeError("Le chargement des données initiales a échoué !")
        if(user_id not in clicsUtilisateur):
            return {"message": f"Utilisateur {user_id} non connu !"}
        #Recommandation
        stats = _getStatsUtilisateur(user_id, clicsUtilisateur)
        modele = _choixModele(stats)
        recommandations = []
        if(modele == "popularity"):
            recommandations = _recommandationsPopularity(user_id, clicsUtilisateur, nbReco)
        elif(modele == "content"):
            recommandations = _recommandationsContent(user_id, clicsUtilisateur, nbReco)
        else: #équivalent modele == "collaborative"
            modeleALS, matrice, user_id_to_index, index_to_article_id = _entrainementCollaboratif(clicsUtilisateur)
            recommandations = _recommandationsCollaborative(user_id, modeleALS, matrice, user_id_to_index, index_to_article_id, nbReco)
        #Métriques et retour
        process = psutil.Process(os.getpid())
        mem_octets = process.memory_info().rss
        mem_mo = mem_octets / (1024 * 1024)
        mem_go = mem_mo / 1024
        duree = round(time.time() - timerStart, 2)
        return {"user_id": int(user_id), "modele_utilise": modele, "statistiques_utilisateur": {k: _convert(v) for k, v in stats.items()}, "recommandations": [{k: _convert(v) for k, v in reco.items()} for reco in recommandations], "memoire_utilisee_GiB": round(mem_go, 2), "temps_execution": duree, "version": "1.3"}
    except Exception as excpt:
        logger.error(f"ERREUR CRITIQUE dans get_recommendations: {excpt}", exc_info=True)
        #On réinitialise les variables globales en cas de souci, pour forcer un rechargement complet
        clicsUtilisateur = articlesDataframe = tableauEmbeddings = article_id_to_index = articleParCategorie = None
        modeleALS = matriceArticles_x_Utilisateurs = user_id_to_index = index_to_article_id = None
        #On retourne l'erreur au log Azure
        return {"erreur": str(excpt)}