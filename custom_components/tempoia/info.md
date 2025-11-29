# TempoIA pour Home Assistant

Intégration complète pour prédire les couleurs Tempo (EDF) directement dans Home Assistant.

## Installation

### Via HACS

1. Ajoutez ce dépôt comme source personnalisée dans HACS
2. Recherchez "TempoIA" dans les intégrations
3. Cliquez sur "Télécharger"
4. Redémarrez Home Assistant

### Manuel

1. Copiez le répertoire `custom_components/tempoia` dans votre dossier `config/custom_components/`
2. Redémarrez Home Assistant

## Configuration

L'intégration nécessite une instance de l'API TempoIA fonctionnelle.

1. Allez dans **Configuration** → **Intégrations**
2. Cliquez sur **+ Ajouter une intégration**
3. Recherchez "TempoIA"
4. Entrez l'URL de votre API et le token (optionnel)

## Fonctionnalités

- ✅ 14 capteurs de prédiction (J+1 à J+14)
- ✅ Calendrier des prédictions
- ✅ Services pour entraîner le modèle et mettre à jour les données
- ✅ Probabilités détaillées pour chaque couleur
- ✅ Support complet des traductions (FR/EN)

## Documentation

Consultez le [README principal](../../README.md) pour plus d'informations sur l'API et le système complet.
