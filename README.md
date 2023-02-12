# Prédiction du prix à la nuit des logements AirBnB à Berlin

Ce projet a été présenté dans le cadre du cours d'Apprentissage automatique de 3ème année du cycle ingénieur CentraleSupélec.


## Rapport 

Le rapport du projet est libre d'accès et se trouve au format `.pdf` dans le répertoire `doc/`.


## Démarrage rapide

Pour exécuter les différents codes du projet, commencer par créer un dossier dédié :

```
mkdir airbnb-price-prediction-project
``` 

Ensuite, récupérer localement le code du dépôt : 

```
cd airbnb-price-prediction-project
git clone https://github.com/bclement1/airbnb-price-prediction
cd airbnb-price-prediction
```

Créer un environnement virtuel dans lequel seront installées les dépendances du projet :

```
python -m venv venv/
source venv/bin/activate
```

Enfin, installer les dépendances du projet dans l'environnement précédent :

```
pip install -r requirements.txt
```

Il ne reste plus qu'à télécharger les données. Commencer par créer un dossier `data/` :

```
mkdir data/
```

Ensuite, rendez-vous sur le site Kaggle pour télécharger le jeu de données qui se trouve à l'adresse suivante :
`https://www.kaggle.com/datasets/gauravduttakiit/airbnb-berlin-price-prediction`. Il faut enregistrer le fichier `.csv` dans le dossier `data/`.


Vous êtes prêt(e) à utiliser le projet !

