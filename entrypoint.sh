#!/bin/sh

# Arrête le script si une commande échoue
set -e

# 1. Préparer la tâche cron
# Récupère le schedule depuis la variable d'environnement, avec une valeur par défaut
CRON_SCHEDULE=${CRON_SCHEDULE:-"20 6 * * *"}

# Commande à exécuter par cron.
# On utilise `python tempoia.py` avec les arguments définis dans les variables d'environnement.
COMMAND_TO_RUN="python tempoia.py --${TEMPOIA_MODE:-auto-mqtt} --forecast ${FORECAST_DAYS:-14} --years ${DB_YEARS:-7} --mqtt-broker ${MQTT_BROKER} --mqtt-port ${MQTT_PORT} --mqtt-topic ${MQTT_TOPIC} --mqtt-user ${MQTT_USER} --mqtt-password ${MQTT_PASSWORD} --mqtt-discovery-prefix ${MQTT_DISCOVERY_PREFIX}"

# Injecte la tâche dans le crontab et ajoute un log pour le suivi
echo "${CRON_SCHEDULE} ${COMMAND_TO_RUN} >> /app/logs/cron.log 2>&1" | crontab -

echo "Tâche Cron configurée : ${CRON_SCHEDULE}"
echo "Commande : ${COMMAND_TO_RUN}"

# 2. Démarrer le service cron en arrière-plan
echo "Démarrage du service cron..."
cron -f &

# 3. Démarrer le serveur API en premier plan (ce qui maintient le conteneur en vie)
echo "Démarrage du serveur API sur le port 8000..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 api:app