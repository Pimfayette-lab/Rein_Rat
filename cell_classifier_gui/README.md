# Cell Classifier GUI

Interface graphique pour le pipeline de classification cellulaire multi-canal
(AQP2/AE1) sur images CZI, basée sur `cell_classifier_2.py`.

## Contenu du dossier

```
cell_classifier_gui/
├── core/
│   ├── cell_classifier_core.py   # moteur d'analyse (logique inchangée)
│   └── __init__.py
├── gui.py                        # interface graphique Tkinter
├── requirements.txt
├── presets/                      # vos presets de paramètres sauvegardés (JSON)
└── README.md
```

Le moteur (`core/cell_classifier_core.py`) reste utilisable en ligne de commande
exactement comme avant. Seule la fonction `run()` a été adaptée pour :
- écrire les résultats dans un dossier de sortie choisi (au lieu du dossier courant) ;
- lever des exceptions Python normales plutôt que d'appeler `sys.exit()`,
  pour que la GUI (ou tout autre appelant) puisse gérer proprement les erreurs.

## Installation

Nécessite Python 3.9+ et un environnement avec affichage graphique (poste de
travail local — cette interface ne fonctionne pas dans un environnement
sans écran).

```bash
cd cell_classifier_gui
python -m venv venv
source venv/bin/activate       # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Note Tkinter : sur Linux, si `tkinter` n'est pas déjà installé avec Python,
installez le paquet système correspondant, par exemple :
```bash
sudo apt install python3-tk
```

## Lancer l'interface

```bash
python gui.py
```

## Utilisation

1. **Fichier CZI** : cliquez sur « Parcourir… » pour choisir l'image à analyser.
2. **Dossier sortie** : choisissez où seront écrits les images annotées et le
   rapport (`annotated_green.png`, `annotated_red.png`, `annotated_blue.png`,
   `annotated_all.png`, `cell_report_v5.txt`).
3. **Paramètres** : tous les champs de la classe `Config` d'origine sont
   modifiables par groupe (Canaux, Détection des noyaux, Canal vert/rouge/bleu,
   Fond local...). Survolez les libellés pour vous repérer — ils reprennent les
   noms des options CLI d'origine.
4. **Presets** : « 💾 Sauver preset… » enregistre tous les paramètres actuels
   (dont le chemin du CZI) dans un fichier JSON réutilisable via
   « 📂 Charger preset… ». Pratique pour garder un jeu de paramètres validé
   pour un type d'acquisition donné.
5. **Auto-entraînement ML** : cochez la case pour activer l'entraînement
   automatique d'un Random Forest à partir des prédictions par seuillage
   (comme l'option `--self-train` du script original). Le modèle est
   sauvegardé/chargé depuis le chemin indiqué.
6. **▶ Lancer l'analyse** : l'analyse tourne en arrière-plan (l'interface reste
   réactive), le journal s'affiche en direct à droite, et un aperçu de
   `annotated_all.png` apparaît automatiquement une fois le traitement terminé.
7. **Ouvrir dossier sortie** : ouvre le dossier de résultats dans l'explorateur
   de fichiers du système.

## Utilisation en ligne de commande (inchangée)

Le moteur reste utilisable directement, comme avant :

```bash
python -m core.cell_classifier_core chemin/vers/image.czi --green-thresh 25
```

## Notes

- Les paramètres exposés dans l'interface correspondent exactement aux champs
  de la `dataclass Config` du script d'origine — aucune logique de détection,
  d'échantillonnage radial, de watershed ou de classification n'a été modifiée.
- Pour de très gros fichiers CZI, le traitement peut prendre plusieurs minutes ;
  le journal affiche la progression (chargement, détection, fond local,
  classification, sauvegarde).
