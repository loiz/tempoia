# Étape 1: Utiliser une image Python légère comme base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Définir des variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installer les dépendances système (si nécessaire, par exemple pour certaines bibliothèques)
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances et les installer
# Cela permet de bénéficier du cache de Docker : les dépendances ne sont réinstallées
# que si le fichier requirements.txt change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Rendre le script d'entrée exécutable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Définir le point d'entrée
ENTRYPOINT ["entrypoint.sh"]