# TempoIA - Prédiction Intelligente des Jours Tempo EDF

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![GitHub Release](https://img.shields.io/github/release/loiz/TempoIA.svg)](https://github.com/loiz/TempoIA/releases)
[![License](https://img.shields.io/github/license/loiz/TempoIA.svg)](LICENSE)

**TempoIA** est un système de prédiction intelligent des couleurs de jours Tempo (EDF) utilisant l'apprentissage automatique et les données météorologiques. Le projet comprend une API FastAPI complète et une intégration Home Assistant native.

## 🌟 Fonctionnalités Principales

- 🤖 **Machine Learning avancé** - Plusieurs algorithmes (MLP, Random Forest, Gradient Boosting, etc.)
- 📊 **Prédictions sur 14 jours** - Anticipez les jours Bleu, Blanc et Rouge
- 🏠 **Intégration Home Assistant** - Installation via HACS, 14 capteurs + calendrier
- 📈 **API REST complète** - Endpoints pour prédictions, statistiques et maintenance
- 🔄 **Mise à jour automatique** - Données Tempo et météo actualisées régulièrement
- 📉 **Statistiques détaillées** - Performances du modèle, cycles Tempo, précision des prédictions

## 🏠 Intégration Home Assistant

### Installation via HACS (Recommandé)

1. Assurez-vous que [HACS](https://hacs.xyz/) est installé
2. Dans HACS, allez dans **Intégrations**
3. Cliquez sur le menu ⋮ → **Dépôts personnalisés**
4. Ajoutez `https://github.com/loiz/TempoIA` comme dépôt de type **Integration**
5. Recherchez "TempoIA" et installez-le
6. Redémarrez Home Assistant
7. Ajoutez l'intégration via **Configuration** → **Intégrations** → **Ajouter**

### Configuration

Renseignez:
- **URL de l'API**: Votre instance TempoIA API (ex: `http://192.168.1.100:8000`)
- **Token API** (optionnel): Token d'authentification si configuré
- **Intervalle de scan**: Fréquence de mise à jour (défaut: 60 minutes)

### Entités Créées

- **14 capteurs** (`sensor.tempoia_jour_1` à `sensor.tempoia_jour_14`) avec probabilités
- **1 calendrier** (`calendar.tempoia_forecast`) pour visualisation
- **3 services**: `train_model`, `update_database`, `refresh_forecast`

[📖 Documentation complète de l'intégration](README_INTEGRATION.md)

## 🚀 API TempoIA

### Déploiement Docker (Recommandé)

```bash
docker run -d \
  --name tempoia-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e API_TOKEN=your-secret-token \
  -e CORS_ORIGINS=* \
  tempoia/api:latest
```

### Installation Locale

```bash
# Cloner le dépôt
git clone https://github.com/loiz/TempoIA.git
cd TempoIA

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict?days=14` | GET | Prédictions multi-jours |
| `/stats/database` | GET | Statistiques de la base de données |
| `/stats/tempo` | GET | Statistiques du cycle Tempo |
| `/stats/model` | GET | Informations sur le modèle |
| `/stats/predictions` | GET | Précision des prédictions |
| `/train` | POST | Entraîner le modèle |
| `/update_database` | POST | Mettre à jour les données |

[📚 Documentation complète de l'API](DOCKER.md)

## 📊 Utilisation en Ligne de Commande

```bash
# Initialiser la base de données (3 ans de données)
python tempoia.py --init-db --years 3

# Entraîner le modèle (sélection automatique du meilleur algorithme)
python tempoia.py --select-algo

# Faire une prédiction
python tempoia.py --forecast 14

# Afficher les statistiques
python tempoia.py --stats
```

### 🔮 Prédiction Personnalisée

Faites des prédictions avec vos propres paramètres :

```bash
python tempoia.py --predict-custom
```

Vous serez invité à entrer :
- Températures (moyenne, max, min)
- Précipitations
- Ensoleillement
- Code météo
- Jours rouges/blancs restants

### 📤 Export CSV

Exportez toutes les données pour analyse externe :

```bash
# Export dans le répertoire courant
python tempoia.py --export-csv

# Export dans un dossier spécifique
python tempoia.py --export-csv --output-dir ./mes_exports
```

Génère 3 fichiers :
- `tempo_data.csv` - Données Tempo
- `weather_data.csv` - Données météo
- `combined_data.csv` - Données combinées

### 🎯 Mode Interactif

Pour une interface menu complète :

```bash
python tempoia.py --interactive
```

Navigation intuitive avec menu numéroté pour accéder à toutes les fonctionnalités.

## Options de Ligne de Commande

| Option | Description |
|--------|-------------|
| `--init-db` | Initialiser la base de données |
| `--update` | Mettre à jour uniquement le cycle en cours (rapide) |
| `--train` | Entraîner le modèle |
| `--predict` | Prédire le jour suivant |
| `--predict-custom` | Prédiction personnalisée |
| `--view-db` | Visualiser toute la base |
| `--view-tempo` | Voir les données Tempo |
| `--view-weather` | Voir les données météo |
| `--view-combined` | Voir les données combinées |
| `--stats` | Afficher les statistiques |
| `--export-csv` | Exporter en CSV |
| `--interactive` | Mode interactif |
| `--limit N` | Limiter l'affichage à N lignes |
| `--years N` | Charger N années de données |
| `--output-dir DIR` | Répertoire d'export |

## Workflow Typique

1. **Premier usage :**
   ```bash
   python tempoia.py --init-db --years 3
   python tempoia.py --train
   python tempoia.py --predict
   ```

2. **Exploration des données :**
   ```bash
   python tempoia.py --stats
   python tempoia.py --view-db --limit 50
   python tempoia.py --export-csv
   ```

3. **Prédictions :**
   ```bash
   # Prédiction standard
   python tempoia.py --predict
   
   # Prédiction avec scénario hypothétique
   python tempoia.py --predict-custom
   ```

4. **Mise à jour périodique :**
   ```bash
   python tempoia.py --update
   python tempoia.py --train
   ```

## Structure de la Base de Données

La base SQLite contient 2 tables :

### `tempo_days`
- `date` - Date du jour
- `color` - Couleur (BLEU, BLANC, ROUGE)
- `red_remaining` - Jours rouges restants
- `white_remaining` - Jours blancs restants
- `cycle_year` - Année du cycle

### `weather_data`
- `date` - Date
- `temperature_avg` - Température moyenne (°C)
- `temperature_max` - Température maximale (°C)
- `temperature_min` - Température minimale (°C)
- `precipitation` - Précipitations (mm)
- `sunshine_duration` - Ensoleillement (heures)
- `weather_code` - Code météo

## Fichiers Générés

- `tempo_weather.db` - Base de données SQLite
- `tempo_model.joblib` - Modèle entraîné
- `scaler.joblib` - Normaliseur de données
- `label_encoder.joblib` - Encodeur de labels
- `*.csv` - Exports CSV (si demandés)

## Aide

Pour voir toutes les options :

```bash
python tempoia.py --help
```

## Notes

- Les couleurs Tempo sont représentées par des emojis : 🔵 BLEU, ⚪ BLANC, 🔴 ROUGE
- Le cycle Tempo court du 1er septembre au 31 août
- Limites annuelles : 22 jours rouges, 43 jours blancs
- Les prédictions sont des probabilités, pas des certitudes
