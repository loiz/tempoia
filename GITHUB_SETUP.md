# 🚀 Guide de Publication sur GitHub

## ✅ Étapes Complétées

1. ✓ Dépôt Git initialisé
2. ✓ Fichier .gitignore créé
3. ✓ Premier commit effectué

## 📋 Étapes Suivantes

### 1. Créer le dépôt sur GitHub

Allez sur **https://github.com/new** et créez un nouveau dépôt :

- **Nom** : `TempoIA`
- **Description** : `Prédiction intelligente des jours Tempo EDF avec IA et intégration Home Assistant`
- **Visibilité** : Public (recommandé pour HACS) ou Privé
- **⚠️ Ne cochez PAS** : "Add a README file", "Add .gitignore", ou "Choose a license"

### 2. Lier le dépôt local à GitHub

Une fois le dépôt créé sur GitHub, exécutez ces commandes **en remplaçant `USERNAME` par votre nom d'utilisateur GitHub** :

```bash
cd /home/loiz/Work/TempoIA

# Changer le nom de la branche principale en 'main' (standard GitHub)
git branch -M main

# Ajouter le dépôt distant (REMPLACEZ 'USERNAME' !)
git remote add origin https://github.com/USERNAME/TempoIA.git

# Pousser le code vers GitHub
git push -u origin main
```

### 3. Vérifier sur GitHub

- Allez sur `https://github.com/USERNAME/TempoIA`
- Vous devriez voir tous vos fichiers !

## 🏠 Activer HACS

Une fois le code sur GitHub, ajoutez le dépôt dans HACS :

1. Dans Home Assistant, allez dans **HACS** → **Intégrations**
2. Menu ⋮ → **Dépôts personnalisés**
3. Ajoutez `https://github.com/USERNAME/TempoIA`
4. Catégorie : **Integration**

## 📦 Créer une Release (Optionnel mais recommandé)

Pour que HACS puisse détecter les versions :

1. Sur GitHub, allez dans **Releases** → **Create a new release**
2. Tag : `v1.1.0`
3. Title : `v1.1.0 - HACS Compatible avec statistiques avancées`
4. Description : Listez les fonctionnalités
5. Cliquez sur **Publish release**

## 🔑 Commandes Git Utiles

```bash
# Voir le statut
git status

# Ajouter des modifications
git add .
git commit -m "Description des changements"
git push

# Créer un tag pour une nouvelle version
git tag v1.1.1
git push --tags
```

## 🆘 Aide

Si vous avez des problèmes d'authentification GitHub, vous aurez besoin d'un **Personal Access Token** :

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. **Generate new token** avec les permissions `repo`
3. Utilisez ce token comme mot de passe quand Git le demande

---

**Note** : Le nom d'utilisateur GitHub dans tous les liens doit être remplacé par votre vrai nom d'utilisateur !
