# Utiliser l'image Python officielle comme base
FROM python:3.12-slim

# Mettre à jour pip, setuptools et wheel
RUN pip install --upgrade pip setuptools wheel

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt requirements.txt
COPY app.py app.py

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel Streamlit sera exécuté
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
