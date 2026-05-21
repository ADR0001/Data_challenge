# Data Challenge - Analyse de sentiments

Ce projet contient un notebook Jupyter qui traite un problème de classification de sentiments à partir de critiques de films. L'objectif est de nettoyer les textes, corriger une partie du bruit présent dans les labels, enrichir le jeu de données, puis entraîner un modèle de classification.

## Contenu du dépôt

- `DATA_CHALLENGE.ipynb` : notebook principal contenant l'ensemble du pipeline d'exploration, de nettoyage, d'augmentation et d'entraînement.

## Objectif du projet

Le notebook cherche à prédire la catégorie de sentiment associée à une critique textuelle. Le travail comprend notamment :

- l'import des jeux de données fournis pour le challenge ;
- le nettoyage des critiques, en retirant le HTML, les caractères spéciaux, les chiffres et les stopwords ;
- la correction orthographique des textes ;
- l'utilisation d'un jeu de données externe IMDB pour améliorer la robustesse du modèle ;
- l'identification et la correction potentielle de labels bruités ;
- l'augmentation de données par remplacement de mots par des synonymes ;
- l'entraînement et l'évaluation de plusieurs modèles de classification.

## Jeux de données attendus

Le notebook utilise plusieurs fichiers CSV :

- `x_baseline (3).csv` : critiques textuelles du jeu d'entraînement ;
- `y_baseline (1).csv` : labels associés aux critiques ;
- `test_ens_sample (4).csv` : jeu de test fourni pour le challenge ;
- `imdb.csv` : jeu de données externe de critiques IMDB utilisé pour enrichir et fiabiliser l'apprentissage.

Attention : les chemins des fichiers sont actuellement codés en dur dans le notebook, par exemple :

```python
X = pd.read_csv("C:\\Users\\Hadder\\Downloads\\x_baseline (3).csv")
df_imdb = pd.read_csv("C:/Users/Hadder/Downloads/imdb.csv")
```

Avant d'exécuter le notebook sur une autre machine, il faut donc adapter ces chemins ou placer les fichiers dans un dossier `data/` et modifier les cellules d'import.

## Pipeline de traitement

Le notebook suit les étapes suivantes :

1. Import des bibliothèques Python nécessaires.
2. Chargement et fusion des données du challenge.
3. Nettoyage du texte :
   - suppression du HTML ;
   - suppression des expressions entre crochets ;
   - suppression des caractères spéciaux ;
   - passage en minuscules ;
   - retrait des stopwords.
4. Visualisation des mots les plus fréquents.
5. Correction orthographique des critiques.
6. Chargement d'un dataset IMDB externe.
7. Entraînement d'un classifieur Naive Bayes avec vectorisation TF-IDF.
8. Test d'un modèle CNN avec Keras/TensorFlow.
9. Correction des labels du jeu initial à partir des prédictions.
10. Suppression des valeurs manquantes et des doublons.
11. Augmentation de données :
    - remplacement par synonymes avec WordNet ;
    - ajout d'un échantillon de données IMDB.
12. Entraînement final avec fastText.
13. Export du fichier final `data_submit.csv`.

## Modèles utilisés

Plusieurs approches sont testées dans le notebook :

- `MultinomialNB` avec `TfidfVectorizer` ;
- un modèle CNN textuel avec `Embedding`, `Conv1D`, `MaxPooling1D` et couches denses ;
- un modèle supervisé `fastText`, utilisé pour l'entraînement final.

## Installation

Il est recommandé d'utiliser un environnement virtuel Python.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn nltk scikit-learn beautifulsoup4 plotly textblob autocorrect tqdm textaugment tensorflow keras keras-preprocessing fasttext pydot
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib seaborn nltk scikit-learn beautifulsoup4 plotly textblob autocorrect tqdm textaugment tensorflow keras keras-preprocessing fasttext pydot
```

Certaines ressources NLTK peuvent être nécessaires :

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

## Utilisation

1. Placer les fichiers CSV nécessaires dans un dossier accessible.
2. Adapter les chemins dans les premières cellules du notebook.
3. Ouvrir le notebook :

```bash
jupyter notebook DATA_CHALLENGE.ipynb
```

4. Exécuter les cellules dans l'ordre.
5. Récupérer le fichier généré `data_submit.csv`.

## Points d'attention

- La correction orthographique peut être longue, car elle parcourt un grand nombre de mots uniques.
- Plusieurs chemins d'entrée et de sortie sont propres à une machine locale et doivent être modifiés avant réexécution.
- Le notebook mélange exploration, nettoyage, entraînement et export. Pour une version production, il serait utile de séparer ces étapes dans des scripts dédiés.
- Le modèle CNN utilise `argmax` sur une sortie sigmoïde, ce qui peut fausser les prédictions binaires. Un seuillage à `0.5` serait plus adapté.
- L'augmentation par synonymes dépend fortement de WordNet et peut parfois modifier le sens initial d'une phrase.

## Résultat attendu

Le notebook produit un jeu de données enrichi et nettoyé, exporté sous le nom :

```text
data_submit.csv
```

Ce fichier peut ensuite être utilisé pour l'entraînement ou la soumission selon les règles du challenge.

