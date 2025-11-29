#!/bin/bash

# Script de scheduler cron pour exécuter tempoia selon les paramètres configurés

# Logs
LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/tempoia_scheduler.log"

# Créer le répertoire logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Fonction de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "====================================================="
log "Démarrage du scheduler TempoIA"
log "====================================================="

# Récupérer les variables d'environnement
CRON_SCHEDULE="${CRON_SCHEDULE:-20 6 * * *}"
TEMPOIA_MODE="${TEMPOIA_MODE:-auto-mqtt}"
MQTT_BROKER="${MQTT_BROKER:-localhost}"
MQTT_PORT="${MQTT_PORT:-1883}"
MQTT_TOPIC="${MQTT_TOPIC:-tempo/forecast}"
MQTT_USER="${MQTT_USER:-}"
MQTT_PASSWORD="${MQTT_PASSWORD:-}"
MQTT_DISCOVERY_PREFIX="${MQTT_DISCOVERY_PREFIX:-homeassistant}"
FORECAST_DAYS="${FORECAST_DAYS:-14}"
DB_YEARS="${DB_YEARS:-3}"
TRAIN_ALGO="${TRAIN_ALGO:-}" # Variable pour l'algorithme d'entraînement

# Option pour exécuter au démarrage
RUN_ON_STARTUP="${RUN_ON_STARTUP:-true}"
log "Configuration du scheduler:"
log "  - Horaire cron: $CRON_SCHEDULE"
log "  - Mode d'exécution: $TEMPOIA_MODE"
log "  - Broker MQTT: $MQTT_BROKER:$MQTT_PORT"
log "  - Topic MQTT: $MQTT_TOPIC"
log "  - Jours de prédiction: $FORECAST_DAYS"
log "  - Années de données: $DB_YEARS"
log "  - Algorithme d'entraînement: ${TRAIN_ALGO:-par défaut}"

# Construire la commande en fonction du mode
case "$TEMPOIA_MODE" in
    "auto-mqtt")
        CRON_CMD="cd /app && python tempoia.py --auto-mqtt \
            --mqtt-broker $MQTT_BROKER \
            --mqtt-port $MQTT_PORT \
            --mqtt-topic $MQTT_TOPIC \
            --mqtt-discovery-prefix $MQTT_DISCOVERY_PREFIX"
        
        if [ -n "$MQTT_USER" ]; then
            CRON_CMD="$CRON_CMD --mqtt-user $MQTT_USER"
        fi
        
        if [ -n "$MQTT_PASSWORD" ]; then
            CRON_CMD="$CRON_CMD --mqtt-password $MQTT_PASSWORD"
        fi
        ;;
    
    "forecast")
        CRON_CMD="cd /app && python tempoia.py --forecast $FORECAST_DAYS"
        ;;
    
    "predict")
        CRON_CMD="cd /app && python tempoia.py --predict"
        ;;
    
    "train")
        CRON_CMD="cd /app && python tempoia.py --train"
        # Ajouter l'argument de l'algorithme si spécifié
        if [ -n "$TRAIN_ALGO" ]; then
            CRON_CMD="$CRON_CMD --algo $TRAIN_ALGO"
        fi
        ;;
    
    "init-db")
        CRON_CMD="cd /app && python tempoia.py --init-db --years $DB_YEARS"
        ;;
    
    "view-db")
        CRON_CMD="cd /app && python tempoia.py --view-db"
        ;;
    
    "export-csv")
        CRON_CMD="cd /app && python tempoia.py --export-csv"
        ;;
    
    *)
        log "ERREUR: Mode inconnu: $TEMPOIA_MODE"
        log "Modes disponibles: auto-mqtt, forecast, predict, train, init-db, view-db, export-csv"
        exit 1
        ;;
esac

# Ajouter la redirection des logs
CRON_CMD="$CRON_CMD >> $LOG_DIR/tempoia_execution.log 2>&1"

# Installation de la tâche cron
echo "$CRON_SCHEDULE $CRON_CMD" | crontab -

log "Tâche cron configurée avec succès"
log "Horaire: $CRON_SCHEDULE"
log "Commande: $(echo $CRON_CMD | head -c 150)..."

# Afficher les tâches cron configurées
log "Tâches cron actuelles:"
crontab -l >> "$LOG_FILE" 2>&1

log "====================================================="
log "Exécution initiale du script au démarrage"
log "====================================================="

# Vérifier si on doit exécuter au démarrage
if [ "$RUN_ON_STARTUP" = "true" ] || [ "$RUN_ON_STARTUP" = "True" ] || [ "$RUN_ON_STARTUP" = "TRUE" ]; then
    log "RUN_ON_STARTUP activé: exécution au démarrage"
    
    # Construire et exécuter la même commande que le cron
    case "$TEMPOIA_MODE" in
        "auto-mqtt")
            STARTUP_CMD="cd /app && python tempoia.py --auto-mqtt \
                --mqtt-broker $MQTT_BROKER \
                --mqtt-port $MQTT_PORT \
                --mqtt-topic $MQTT_TOPIC \
                --mqtt-discovery-prefix $MQTT_DISCOVERY_PREFIX"
            
            if [ -n "$MQTT_USER" ]; then
                STARTUP_CMD="$STARTUP_CMD --mqtt-user $MQTT_USER"
            fi
            
            if [ -n "$MQTT_PASSWORD" ]; then
                STARTUP_CMD="$STARTUP_CMD --mqtt-password $MQTT_PASSWORD"
            fi
            ;;
        
        "forecast")
            STARTUP_CMD="cd /app && python tempoia.py --forecast $FORECAST_DAYS"
            ;;
        
        "predict")
            STARTUP_CMD="cd /app && python tempoia.py --predict"
            ;;
        
        "train")
            STARTUP_CMD="cd /app && python tempoia.py --train"
            # Ajouter l'argument de l'algorithme si spécifié
            if [ -n "$TRAIN_ALGO" ]; then
                STARTUP_CMD="$STARTUP_CMD --algo $TRAIN_ALGO"
            fi
            ;;
        
        "init-db")
            STARTUP_CMD="cd /app && python tempoia.py --init-db --years $DB_YEARS"
            ;;
        
        "view-db")
            STARTUP_CMD="cd /app && python tempoia.py --view-db"
            ;;
        
        "export-csv")
            STARTUP_CMD="cd /app && python tempoia.py --export-csv"
            ;;
        
        *)
            log "ERREUR: Mode inconnu: $TEMPOIA_MODE"
            log "Modes disponibles: auto-mqtt, forecast, predict, train, init-db, view-db, export-csv"
            exit 1
            ;;
    esac
    
    log "Commande: $STARTUP_CMD"
    log "====================================================="
    
    # Exécuter la commande au démarrage
    eval "$STARTUP_CMD" >> "$LOG_DIR/tempoia_startup.log" 2>&1
    STARTUP_EXIT_CODE=$?
    
    if [ $STARTUP_EXIT_CODE -eq 0 ]; then
        log "✅ Exécution au démarrage terminée avec succès"
    else
        log "⚠️  Exécution au démarrage terminée avec le code: $STARTUP_EXIT_CODE"
    fi
else
    log "RUN_ON_STARTUP désactivé: pas d'exécution au démarrage"
fi

log "====================================================="
log "Lancement de cron en mode foreground"
log "====================================================="

# Démarrer cron en avant-plan
exec "$@"
