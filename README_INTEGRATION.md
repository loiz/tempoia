# TempoIA - Prédiction de Couleurs Tempo pour Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)

Intégration Home Assistant pour prédire les couleurs des jours Tempo (EDF) en France en utilisant l'intelligence artificielle et les prévisions météorologiques.

## Fonctionnalités

✨ **Prédictions sur 14 jours** - Visualisez les prédictions de couleur Tempo pour les 2 prochaines semaines  
🤖 **IA avancée** - Utilise un modèle d'apprentissage automatique entraîné sur les données historiques  
📊 **Probabilités détaillées** - Obtenez les probabilités pour chaque couleur (Bleu, Blanc, Rouge)  
📅 **Entité calendrier** - Intégration calendrier pour visualiser les prédictions  
🔄 **Services HA** - Entraînez le modèle et mettez à jour la base de données depuis Home Assistant  
🌐 **API externe** - Nécessite une instance API TempoIA en cours d'exécution

## Prérequis

- Home Assistant 2023.1 ou supérieur
- Une instance de l'API TempoIA en cours d'exécution (voir [TempoIA API](https://github.com/loiz/TempoIA))

## Installation

### Via HACS (Recommandé)

1. Assurez-vous que [HACS](https://hacs.xyz/) est installé
2. Dans HACS, allez dans "Intégrations"
3. Cliquez sur le menu ⋮ en haut à droite
4. Sélectionnez "Dépôts personnalisés"
5. Ajoutez `https://github.com/loiz/TempoIA` comme dépôt de type "Integration"
6. Recherchez "TempoIA" et installez-le
7. Redémarrez Home Assistant

###  Installation Manuelle

1. Téléchargez la dernière version depuis [Releases](https://github.com/loiz/TempoIA/releases)
2. Copiez le dossier `custom_components/tempoia` dans `<config>/custom_components/`
3. Redémarrez Home Assistant

## Configuration

### Via l'Interface Utilisateur

1. Allez dans **Configuration** → **Appareils & Services**
2. Cliquez sur **+ Ajouter une intégration**
3. Recherchez "TempoIA"
4. Renseignez:
   - **URL de l'API**: L'URL de votre API TempoIA (ex: `http://192.168.1.100:8000`)
   - **Token API** (optionnel): Votre token d'API si l'authentification est activée
   - **Intervalle de scan** (optionnel): Fréquence de mise à jour en minutes (défaut: 60)

### Configuration de l'API

Assurez-vous que votre API TempoIA est accessible depuis Home Assistant. Vous pouvez la déployer via Docker:

```bash
docker run -d \\
  --name tempoia-api \\
  -p 8000:8000 \\
  -v $(pwd)/data:/app/data \\
  -e API_TOKEN=your-secret-token \\
  tempoia/api:latest
```

## Entités Créées

### Capteurs (14 entités)

L'intégration crée 14 capteurs, un pour chaque jour de prédiction:

- `sensor.tempoia_jour_1` - Prédiction pour demain (J+1)
- `sensor.tempoia_jour_2` - Prédiction pour J+2
- ...
- `sensor.tempoia_jour_14` - Prédiction pour J+14

Chaque capteur affiche un emoji représentant la couleur prédite:
- 🔵 Bleu
- ⚪ Blanc
- 🔴 Rouge

#### Attributs des Capteurs

```yaml
date: "2025-11-30"
jour: "Samedi"
proba_bleu: 0.75
proba_blanc: 0.20
proba_rouge: 0.05
```

### Calendrier

- `calendar.tempoia_forecast` - Calendrier affichant les prédictions pour les 14 prochains jours

## Services

### `tempoia.train_model`

Déclenche l'entraînement du modèle sur l'API TempoIA.

```yaml
service: tempoia.train_model
```

### `tempoia.update_database`

Met à jour la base de données avec les dernières données Tempo et météo.

```yaml
service: tempoia.update_database
data:
  years: 10  # Nombre d'années de données à récupérer (optionnel, défaut: 10)
```

### `tempoia.refresh_forecast`

Force une mise à jour immédiate des prédictions.

```yaml
service: tempoia.refresh_forecast
```

## Exemples d'Utilisation

### Automatisation - Notification Jour Rouge

```yaml
automation:
  - alias: "Notification Jour Rouge Demain"
    trigger:
      - platform: state
        entity_id: sensor.tempoia_jour_1
        to: "🔴"
    action:
      - service: notify.mobile_app
        data:
          title: "⚡ Jour Rouge Demain"
          message: "Pensez à limiter votre consommation électrique demain!"
```

### Carte Dashboard

```yaml
type: entities
title: Prédictions Tempo
entities:
  - entity: sensor.tempoia_jour_1
  - entity: sensor.tempoia_jour_2
  - entity: sensor.tempoia_jour_3
  - entity: sensor.tempoia_jour_4
  - entity: sensor.tempoia_jour_5
```

### Carte Calendrier

```yaml
type: calendar
entities:
  - calendar.tempoia_forecast
```

## FAQ

### Les prédictions ne se mettent pas à jour

1. Vérifiez que l'API est accessible depuis Home Assistant
2. Vérifiez les logs de Home Assistant: **Configuration** → **Logs**
3. Essayez d'appeler le service `tempoia.refresh_forecast`

### Les probabilités sont toutes à zéro

Cela peut arriver si:
- Le modèle n'a pas été entraîné sur l'API
- Les données météo ne sont pas disponibles
- Appelez le service `tempoia.update_database` puis `tempoia.train_model`

### Comment améliorer les prédictions?

1. Assurez-vous d'avoir au moins 3 ans de données historiques
2. Entraînez régulièrement le modèle (une fois par mois recommandé)
3. Mettez à jour la base de données hebdomadairement

## Support

- 🐛 [Signaler un bug](https://github.com/loiz/TempoIA/issues)
- 💬 [Discussions](https://github.com/loiz/TempoIA/discussions)
- 📖 [Documentation complète](https://github.com/loiz/TempoIA/wiki)

## Licence

MIT License - voir [LICENSE](LICENSE)

## Crédits

- Données Tempo: [API Couleur Tempo](https://www.api-couleur-tempo.fr/)
- Données Météo: [Open-Meteo](https://open-meteo.com/)
- Développé par [@loiz](https://github.com/loiz)
