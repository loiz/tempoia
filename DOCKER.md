# TempoIA - Docker Configuration

## Structure des fichiers

```
TempoIA/
├── Dockerfile              # Image Docker
├── docker-compose.yml      # Orchestration des conteneurs
├── docker-entrypoint.sh    # Script de démarrage avec cron
├── requirements.txt        # Dépendances Python
├── .env.example            # Exemple de fichier d'env
├── .env                    # Configuration (à créer)
├── tempoia.py             # Script principal
├── tempo_weather.db       # Base de données (persistante)
├── tempo_model.joblib     # Modèle ML (persistant)
├── scaler.joblib          # Scaler (persistant)
├── label_encoder.joblib   # Encodeur (persistant)
├── data/                  # Dossier des données
├── logs/                  # Dossier des logs
└── mqtt-config/           # Configuration MQTT (optionnel)
```

## Installation & Démarrage

### 1. Créer le fichier de configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer le fichier selon vos besoins
nano .env
```

### 2. Construction et démarrage du conteneur

```bash
# Construction
docker-compose build

# Démarrage
docker-compose up -d

# Vérification de l'état
docker-compose ps
```

### 3. Vérification des logs

```bash
# Logs du scheduler
docker-compose logs -f tempoia | tail -50

# Logs d'exécution du script
docker exec tempoia-predictor tail -f /app/logs/tempoia_execution.log

# Logs du scheduler cron
docker exec tempoia-predictor tail -f /app/logs/tempoia_scheduler.log
```

### 4. Vérification du cron

```bash
# Voir la tâche cron configurée
docker exec tempoia-predictor crontab -l

# Tester l'exécution manuelle
docker exec tempoia-predictor python tempoia.py --auto-mqtt
```

## Configuration via Variables d'Environnement

Le fichier `.env` contrôle complètement le comportement du conteneur.

### Variables disponibles

#### 🕐 Planification (Cron)

```env
# Format cron: minute hour day month day-of-week
CRON_SCHEDULE=20 6 * * *

# Exécuter au démarrage du conteneur (true/false)
RUN_ON_STARTUP=true
```

**Explications:**
- `CRON_SCHEDULE` - Définit l'horaire de l'exécution planifiée (voir exemples ci-dessous)
- `RUN_ON_STARTUP` - Si `true`, le script s'exécute **immédiatement au démarrage**, puis selon le cron

**Exemples de CRON_SCHEDULE:**
- `20 6 * * *` = 6h20 chaque jour (défaut)
- `0 6 * * *` = 6h00 chaque jour
- `0 */6 * * *` = Toutes les 6 heures
- `*/30 * * * *` = Toutes les 30 minutes
- `0 1 * * 0` = Chaque dimanche à 1h00
- `0 0 1 * *` = Le 1er du mois à minuit

#### 🔧 Mode d'Exécution

```env
# Mode d'exécution
TEMPOIA_MODE=auto-mqtt
```

**Modes disponibles:**
- `auto-mqtt` - Init DB + Train + Forecast + MQTT (défaut)
- `forecast` - Prédiction sur N jours
- `predict` - Prédiction jour suivant
- `train` - Entraîner le modèle
- `init-db` - Initialiser la base de données
- `view-db` - Afficher les données
- `export-csv` - Exporter en CSV

#### 📡 Configuration MQTT

```env
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC=tempo/forecast
MQTT_USER=
MQTT_PASSWORD=
MQTT_DISCOVERY_PREFIX=homeassistant
```

#### 📊 Paramètres de Prédiction

```env
FORECAST_DAYS=14      # Nombre de jours à prédire
DB_YEARS=3            # Années de données à charger
```

## Exemples d'Utilisation

### Exemple 1: Exécution quotidienne à 6h20 (défaut)

```env
CRON_SCHEDULE=20 6 * * *
RUN_ON_STARTUP=true
TEMPOIA_MODE=auto-mqtt
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC=tempo/forecast
```

**Comportement:**
- Au démarrage du conteneur: exécution immédiate
- Puis chaque jour à 6h20

```bash
docker-compose up -d
```

### Exemple 2: Exécution toutes les heures + au démarrage

```env
CRON_SCHEDULE=0 * * * *
RUN_ON_STARTUP=true
TEMPOIA_MODE=forecast
FORECAST_DAYS=9
```

**Comportement:**
- Au démarrage du conteneur: exécution immédiate
- Puis toutes les heures à la minute 0

### Exemple 3: MQTT avec authentification

```env
RUN_ON_STARTUP=true
TEMPOIA_MODE=auto-mqtt
MQTT_BROKER=mqtt.example.com
MQTT_PORT=8883
MQTT_USER=mon_user
MQTT_PASSWORD=mon_password
MQTT_TOPIC=domotique/tempo
```

### Exemple 4: Initialiser la base et entraîner (une fois au démarrage)

```env
RUN_ON_STARTUP=true
CRON_SCHEDULE=0 2 * * *    # 2h du matin (pour les réentraînements)
TEMPOIA_MODE=init-db
DB_YEARS=5
```

Puis après initialisation, changer en:
```env
TEMPOIA_MODE=train
```

### Exemple 5: Export CSV hebdomadaire + au démarrage

```env
RUN_ON_STARTUP=true
CRON_SCHEDULE=0 3 * * 0    # Dimanche 3h du matin
TEMPOIA_MODE=export-csv
```

### Exemple 6: Pas d'exécution au démarrage (cron seulement)

```env
RUN_ON_STARTUP=false
CRON_SCHEDULE=0 6 * * *
TEMPOIA_MODE=predict
```

**Comportement:**
- Au démarrage du conteneur: rien ne s'exécute
- Ensuite chaque jour à 6h00

## Modification de la Configuration

### Modifier l'horaire d'exécution

```bash
# Éditer le fichier .env
nano .env

# Changer CRON_SCHEDULE
CRON_SCHEDULE=0 6 * * *    # Passer à 6h00

# Redémarrer le conteneur
docker-compose restart tempoia
```

### Activer/Désactiver l'exécution au démarrage

```bash
nano .env

# Activer (défaut)
RUN_ON_STARTUP=true

# Désactiver
RUN_ON_STARTUP=false

docker-compose restart tempoia
```

### Changer le mode d'exécution

```bash
nano .env
TEMPOIA_MODE=forecast
FORECAST_DAYS=9

docker-compose restart tempoia
```

### Modifier les paramètres MQTT

```bash
nano .env
MQTT_BROKER=mon-broker.local
MQTT_USER=admin
MQTT_PASSWORD=secret123

docker-compose restart tempoia
```

## Avec MQTT (optionnel)

### Option 1: MQTT local (inclus dans docker-compose)

```bash
# Démarrer avec le profil MQTT
docker-compose --profile mqtt up -d

# Dans le fichier .env
MQTT_BROKER=mqtt
MQTT_PORT=1883
```

### Option 2: MQTT externe

```env
# Dans .env
MQTT_BROKER=mon-broker.example.com
MQTT_PORT=1883
MQTT_USER=mon_user
MQTT_PASSWORD=mon_password
```

### Intégration Home Assistant

Les données sont publiées automatiquement avec Home Assistant Discovery:
- Topics: `homeassistant/sensor/tempo_ia/day_1/config`
- Ajoute automatiquement les capteurs à HA

## Commandes Utiles

```bash
# Démarrage en arrière-plan
docker-compose up -d

# Arrêt du conteneur
docker-compose down

# Reconstruction après modification
docker-compose build --no-cache
docker-compose up -d

# Logs en temps réel
docker-compose logs -f

# Logs du démarrage
docker exec tempoia-predictor cat /app/logs/tempoia_startup.log

# Logs du scheduler
docker exec tempoia-predictor cat /app/logs/tempoia_scheduler.log

# Logs d'exécution (dernières 100 lignes)
docker exec tempoia-predictor tail -100 /app/logs/tempoia_execution.log

# Exécution manuelle directe (ignore le cron)
docker exec tempoia-predictor python tempoia.py --predict

# Entrée dans le conteneur
docker exec -it tempoia-predictor bash

# Voir les tâches cron
docker exec tempoia-predictor crontab -l

# Vérifier l'état de la base de données
docker exec tempoia-predictor python tempoia.py --view-db --limit 10

# Nettoyage complet (attention: supprime les données!)
docker-compose down -v
```

## Ressources

- **CPU** : Limité à 1 cœur (réserve: 0.5)
- **Mémoire** : 512 MB max (réserve: 256 MB)
- **Stockage** : Dépend de la taille de la base de données

## Troubleshooting

### Le cron ne s'exécute pas

1. Vérifiez le fichier `.env`:
   ```bash
   cat .env
   ```

2. Vérifiez les logs du scheduler:
   ```bash
   docker exec tempoia-predictor cat /app/logs/tempoia_scheduler.log
   ```

3. Redémarrez le conteneur:
   ```bash
   docker-compose restart tempoia
   ```

### Erreur "Mode inconnu"

Vérifiez la variable `TEMPOIA_MODE` dans `.env`:
```bash
grep TEMPOIA_MODE .env

# Modes valides: auto-mqtt, forecast, predict, train, init-db, view-db, export-csv
```

### Erreur de connexion MQTT

```bash
# Testez la connexion au broker
docker exec tempoia-predictor nc -zv mqtt 1883

# Vérifiez les logs MQTT
docker logs tempoia-mqtt
```

### Pas de fichier `.env`

Le conteneur utilise les valeurs par défaut. Pour personnaliser:
```bash
cp .env.example .env
nano .env
docker-compose up -d
```

## Workflow Recommandé

### Installation initiale

```bash
# 1. Copier la configuration
cp .env.example .env

# 2. Éditer selon vos besoins
nano .env

# 3. Construire et démarrer
docker-compose build
docker-compose up -d

# 4. Vérifier les logs du démarrage
docker-compose logs -f

# 5. Vérifier les logs d'exécution au démarrage
docker exec tempoia-predictor cat /app/logs/tempoia_startup.log

# 6. Attendre la première exécution planifiée
```

### Mise à jour de la configuration

```bash
# 1. Modifier le fichier .env
nano .env

# 2. Redémarrer le conteneur
docker-compose restart tempoia

# 3. Vérifier les logs
docker-compose logs -f
```

## Notes Importantes

⚠️ **Variables d'environnement vides**: Laissez les champs MQTT_USER et MQTT_PASSWORD vides si pas d'authentification

⚠️ **RUN_ON_STARTUP**: Par défaut à `true` - le script s'exécute immédiatement au démarrage du conteneur

⚠️ **Sensibilité au format cron**: Respectez le format `minute hour day month day-of-week`

⚠️ **Zone horaire**: Par défaut `Europe/Paris` (configurable dans docker-compose.yml)

⚠️ **Première exécution**: Peut prendre 5-10 minutes pour initialiser la BD

⚠️ **Logs multiples**: 
- `tempoia_scheduler.log` - Logs du scheduler cron
- `tempoia_startup.log` - Logs de l'exécution au démarrage
- `tempoia_execution.log` - Logs des exécutions planifiées

⚠️ **Backup**: Sauvegardez régulièrement les fichiers `.db` et `.joblib`

## Support

Pour plus d'informations:
```bash
# Aide du script Python
docker exec tempoia-predictor python tempoia.py --help

# Documentation README
cat README.md
```
