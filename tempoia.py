import sqlite3
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

class TempoWeatherPredictor:
    def __init__(self, db_path='tempo_weather.db'):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.create_database()
    
    def create_database(self):
        """Crée la structure de la base de données SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des jours Tempo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tempo_days (
                date TEXT PRIMARY KEY,
                color TEXT,
                red_remaining INTEGER,
                white_remaining INTEGER,
                cycle_year INTEGER,
                day_of_week INTEGER
            )
        ''')
        
        # Table des données météo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                date TEXT PRIMARY KEY,
                temperature_avg REAL,
                temperature_max REAL,
                temperature_min REAL,
                precipitation REAL,
                sunshine_duration REAL,
                weather_code INTEGER,
                cloud_cover REAL,
                wind_speed REAL,
                FOREIGN KEY (date) REFERENCES tempo_days (date)
            )
        ''')
        
        # Table pour l'historique des entraînements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_date TEXT NOT NULL,
                model_algorithm TEXT NOT NULL,
                n_samples INTEGER,
                test_accuracy REAL,
                test_f1_macro REAL,
                class_report_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    
    def fetch_tempo_data(self, year=None):
        """Récupère les données Tempo depuis l'API"""
        if year is None:
            year = datetime.now().year
        
        base_url = "https://www.api-couleur-tempo.fr/api/joursTempo"
        
        # Définition du cycle (1er septembre au 31 août)
        start_date = f"{year-1}-09-01"
        end_date = f"{year}-08-31"
        
        print(f"Récupération des données Tempo pour le cycle {year-1}-{year}...")
        
        # Génération de toutes les dates du cycle
        dates = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_date_obj:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        all_tempo_data = []
        
        # Récupération par lots de 30 jours
        for i in range(0, len(dates), 30):
            batch_dates = dates[i:i+30]
            date_params = [f"dateJour[]={date}" for date in batch_dates]
            url = f"{base_url}?{'&'.join(date_params)}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                batch_data = response.json()
                
                # Convertir les codes jour en format cohérent
                for day in batch_data:
                    if 'codeJour' in day:
                        # S'assurer que le code est un entier
                        day['codeJour'] = int(day['codeJour'])
                
                all_tempo_data.extend(batch_data)
                print(f"Récupération des jours {i+1} à {i+len(batch_dates)}")
            except requests.RequestException as e:
                print(f"Erreur lors de la récupération: {e}")
                # Simulation de données en cas d'échec
                #all_tempo_data.extend(self._simulate_tempo_data(batch_dates))
        
        return all_tempo_data
    
    
    def calculate_remaining_days(self, tempo_data, year):
        """Calcule le nombre de jours rouges et blancs restants de manière optimisée."""
        red_total = 22
        white_total = 43
        
        # Créer un dictionnaire pour un accès rapide par date (O(N))
        data_by_date = {day['dateJour']: day for day in tempo_data}
        
        # Générer la séquence de dates triées pour le cycle (O(N))
        start_date = datetime.strptime(f"{year-1}-09-01", "%Y-%m-%d")
        end_date = datetime.strptime(f"{year}-08-31", "%Y-%m-%d")
        
        date_sequence = []
        current_date = start_date
        while current_date <= end_date:
            date_sequence.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        current_red = red_total
        current_white = white_total
        
        # Itérer sur la séquence de dates triées (O(N))
        for date_str in date_sequence:
            if date_str in data_by_date:
                day = data_by_date[date_str]
                color = day['codeJour']
                
                if color == 2:  # Blanc
                    current_white -= 1
                elif color == 3:  # Rouge
                    current_red -= 1
                
                day['red_remaining'] = max(0, current_white)
                day['white_remaining'] = max(0, current_red)
                day['cycle_year'] = year
        
        return tempo_data
    
    def insert_tempo_data(self, tempo_data):
        """Insère les données Tempo dans la base SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        violations = {'weekend_red': 0, 'sunday_not_blue': 0}
        
        for day in tempo_data:
            # Conversion du code couleur en chaîne cohérente
            color_map = {1: "BLEU", 2: "BLANC", 3: "ROUGE"}
            color_str = color_map.get(day['codeJour'], "BLEU")
            
            # Calcul du jour de la semaine (0=Lundi, 6=Dimanche)
            date_obj = datetime.strptime(day['dateJour'], "%Y-%m-%d")
            day_of_week = date_obj.weekday()  # 0=Lundi, 6=Dimanche
            
            # Validation des règles Tempo
            if color_str == "ROUGE" and day_of_week >= 5:  # Samedi (5) ou Dimanche (6)
                violations['weekend_red'] += 1
                print(f"⚠️  Violation règle Tempo: Jour ROUGE un {['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week]} ({day['dateJour']})")
            
            if day_of_week == 6 and color_str != "BLEU":  # Dimanche
                violations['sunday_not_blue'] += 1
                print(f"⚠️  Violation règle Tempo: Dimanche {color_str} au lieu de BLEU ({day['dateJour']})")
            
            cursor.execute('''
                INSERT OR REPLACE INTO tempo_days 
                (date, color, red_remaining, white_remaining, cycle_year, day_of_week)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                day['dateJour'],
                color_str,
                day.get('red_remaining', 22),
                day.get('white_remaining', 43),
                day.get('cycle_year', datetime.now().year),
                day_of_week
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Insertion de {len(tempo_data)} jours Tempo")
        if violations['weekend_red'] > 0 or violations['sunday_not_blue'] > 0:
            print(f"⚠️  Violations détectées: {violations['weekend_red']} jours rouges en weekend, {violations['sunday_not_blue']} dimanches non-bleus")

    
    def fetch_historical_weather(self, start_date, end_date, lat=48.8566, lon=2.3522):
        """
        Récupère les données météo historiques depuis Open-Meteo API
        Coordonnées par défaut : Paris
        """
        print(f"Récupération des données météo du {start_date} au {end_date}...")
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 
                     'precipitation_sum', 'sunshine_duration', 'weather_code', 'cloudcover_mean',
                     'wind_speed_10m_max'],
            'timezone': 'Europe/Paris'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            weather_data = []
            daily_data = data.get('daily', {})
            
            for i in range(len(daily_data.get('time', []))):
                # Gestion des valeurs None
                sunshine_duration = daily_data['sunshine_duration'][i] or 0
                sunshine_hours = sunshine_duration / 3600 if sunshine_duration > 0 else 0
                
                weather_data.append({
                    'date': daily_data['time'][i],
                    'temperature_max': daily_data['temperature_2m_max'][i] or 0,
                    'temperature_min': daily_data['temperature_2m_min'][i] or 0,
                    'temperature_avg': daily_data['temperature_2m_mean'][i] or 0,
                    'precipitation': daily_data['precipitation_sum'][i] or 0,
                    'sunshine_duration': round(sunshine_hours, 1),
                    'weather_code': daily_data['weather_code'][i] or 0,
                    'cloud_cover': daily_data['cloudcover_mean'][i] or 0,
                    'wind_speed': daily_data['wind_speed_10m_max'][i] or 0
                })
            
            print(f"✅ Récupération de {len(weather_data)} jours de données météo")
            return weather_data
            
        except requests.RequestException as e:
            print(f"❌ Erreur API météo: {e}")
            print("Utilisation de données météo simulées...")
            return self._simulate_weather_data(start_date, end_date)
            
    
    def fetch_weather_forecast(self, days=9, lat=48.8566, lon=2.3522):
        """
        Récupère les prévisions météo pour les N prochains jours
        """
        print(f"Récupération des prévisions météo pour les {days} prochains jours...")
        
        base_url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 
                     'precipitation_sum', 'sunshine_duration', 'weather_code', 'cloudcover_mean',
                     'wind_speed_10m_max'],
            'timezone': 'Europe/Paris',
            'forecast_days': min(days, 16)
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            weather_data = []
            daily_data = data.get('daily', {})
            
            for i in range(len(daily_data.get('time', []))):
                # Gestion des valeurs None
                sunshine_duration = daily_data['sunshine_duration'][i] or 0
                sunshine_hours = sunshine_duration / 3600 if sunshine_duration > 0 else 0
                
                weather_data.append({
                    'date': daily_data['time'][i],
                    'temperature_max': daily_data['temperature_2m_max'][i] or 0,
                    'temperature_min': daily_data['temperature_2m_min'][i] or 0,
                    'temperature_avg': daily_data['temperature_2m_mean'][i] or 0,
                    'precipitation': daily_data['precipitation_sum'][i] or 0,
                    'sunshine_duration': round(sunshine_hours, 1),
                    'weather_code': daily_data['weather_code'][i] or 0,
                    'cloud_cover': daily_data['cloudcover_mean'][i] or 0,
                    'wind_speed': daily_data['wind_speed_10m_max'][i] or 0
                })
            
            print(f"✅ Récupération de {len(weather_data)} jours de prévisions")
            return weather_data
            
        except requests.RequestException as e:
            print(f"❌ Erreur API météo (Forecast): {e}")
            return None

    def fetch_tempo_stats(self):
        """
        Récupère les statistiques Tempo (jours restants) depuis l'API externe
        """
        url = "https://www.api-couleur-tempo.fr/api/stats"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            red_remaining = data.get('joursRougesRestants')
            white_remaining = data.get('joursBlancsRestants')
            
            if red_remaining is not None and white_remaining is not None:
                print(f"✅ Stats Tempo récupérées: {red_remaining} rouges, {white_remaining} blancs restants")
                return red_remaining, white_remaining
            else:
                print("⚠️  Données manquantes dans la réponse API Stats")
                return None
                
        except requests.RequestException as e:
            print(f"⚠️  Impossible de récupérer les stats Tempo: {e}")
            return None

    def _simulate_weather_data(self, start_date, end_date):
        """Simule des données météo si l'API échoue"""
        simulated_data = []
        
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_date_obj:
            month = current_date.month
            
            # Simulation basée sur la saison
            if month in [12, 1, 2]:  # Hiver
                temp_avg = np.random.uniform(2, 8)
                temp_max = temp_avg + np.random.uniform(2, 5)
                temp_min = temp_avg - np.random.uniform(2, 5)
                sunshine = np.random.uniform(1, 4)
            elif month in [6, 7, 8]:  # Été
                temp_avg = np.random.uniform(18, 25)
                temp_max = temp_avg + np.random.uniform(3, 8)
                temp_min = temp_avg - np.random.uniform(2, 5)
                sunshine = np.random.uniform(6, 12)
            else:  # Printemps/Automne
                temp_avg = np.random.uniform(10, 16)
                temp_max = temp_avg + np.random.uniform(2, 6)
                temp_min = temp_avg - np.random.uniform(2, 5)
                sunshine = np.random.uniform(3, 7)
            
            simulated_data.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'temperature_avg': round(temp_avg, 1),
                'temperature_max': round(temp_max, 1),
                'temperature_min': round(temp_min, 1),
                'precipitation': round(np.random.uniform(0, 5), 1),
                'sunshine_duration': round(sunshine, 1),
                'weather_code': np.random.choice([0, 1, 2, 3, 45, 51, 61, 80]),
                'cloud_cover': round(np.random.uniform(20, 80), 1),  # Couverture nuageuse en %
                'wind_speed': round(np.random.uniform(0, 100), 1)    # Vitesse vent en km/h
            })
            
            current_date += timedelta(days=1)
        
        return simulated_data
    
    def insert_weather_data(self, weather_data):
        """Insère les données météo dans la base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for weather in weather_data:
            cursor.execute('''
                INSERT OR REPLACE INTO weather_data 
                (date, temperature_avg, temperature_max, temperature_min, 
                 precipitation, sunshine_duration, weather_code, cloud_cover, wind_speed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                weather['date'],
                weather['temperature_avg'],
                weather['temperature_max'],
                weather['temperature_min'],
                weather['precipitation'],
                weather['sunshine_duration'],
                weather['weather_code'],
                weather.get('cloud_cover', 50),
                weather.get('wind_speed', 15)  # Valeur par défaut
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Insertion de {len(weather_data)} jours de données météo")
    
    def prepare_training_data(self):
        """Prépare les données pour l'entraînement du réseau de neurones"""
        conn = sqlite3.connect(self.db_path)
        
        # Jointure des données Tempo et météo
        query = '''
            SELECT 
                w.temperature_avg,
                w.temperature_max,
                w.temperature_min,
                w.precipitation,
                w.sunshine_duration,
                w.weather_code,
                w.cloud_cover,
                w.wind_speed,
                t.red_remaining,
                t.white_remaining,
                t.day_of_week,
                t.color as target_color
            FROM weather_data w
            JOIN tempo_days t ON w.date = t.date
            WHERE t.color IS NOT NULL
            ORDER BY w.date
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            raise ValueError("Aucune donnée d'entraînement disponible")
        
        # Nettoyage des données
        df = df.dropna()
        
        # Conversion des colonnes en types numériques
        numeric_columns = ['temperature_avg', 'temperature_max', 'temperature_min', 
                          'precipitation', 'sunshine_duration', 'weather_code', 'cloud_cover', 'wind_speed',
                          'red_remaining', 'white_remaining', 'day_of_week']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Suppression des lignes avec des valeurs NaN après conversion
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Aucune donnée valide après nettoyage")
        
        # Préparation des features et target
        feature_columns = ['temperature_avg', 'temperature_max', 'temperature_min', 
                          'precipitation', 'sunshine_duration', 'weather_code', 'cloud_cover', 'wind_speed',
                          'red_remaining', 'white_remaining', 'day_of_week']
        
        features = df[feature_columns]
        target = df['target_color']
        
        # Encodage des labels cibles
        target_encoded = self.label_encoder.fit_transform(target)
        
        # Normalisation des features
        features_scaled = self.scaler.fit_transform(features)
        
        print(f"📊 Données préparées: {len(features)} échantillons, {len(feature_columns)} features")
        print(f"   Features: température (moy/max/min), précipitations, soleil, code météo, couverture nuageuse,")
        print(f"             vitesse vent, jours restants (rouge/blanc), jour de la semaine")
        print(f"🎯 Classes cibles: {list(self.label_encoder.classes_)}")
        
        return features_scaled, target_encoded, feature_columns
    
    def train_model(self, estimator_key=None):
        """Entraîne le réseau de neurones"""
        print("🔄 Préparation des données d'entraînement...")

        try:
            features, target, feature_names = self.prepare_training_data()
        except ValueError as e:
            print(f"❌ {e}")
            return False
        
            print(f"📊 {len(features)} échantillons d'entraînement disponibles")
        
        # Vérification des types de données
        print(f"🔍 Type des features: {type(features)}, forme: {features.shape}")
        print(f"🔍 Type de target: {type(target)}, forme: {target.shape}")
        print(f"🔍 Type des éléments features: {features.dtype}")
        print(f"🔍 Type des éléments target: {target.dtype}")
        
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        # Choisir l'estimateur en fonction de estimator_key
        estimator = None
        if estimator_key:
            alg_map = self.get_algorithm_map()
            if estimator_key not in alg_map:
                print(f"❌ Algorithme inconnu: {estimator_key}. Clés valides: {list(alg_map.keys())}")
                return False
            estimator = alg_map[estimator_key]
            print(f"🧠 Entraînement de l'algorithme forcé: {estimator_key} -> {estimator.__class__.__name__}")
        else:
            # par défaut, le MLP déjà utilisé précédemment
            estimator = MLPClassifier(
                hidden_layer_sizes=(32, 16),  # Réseau plus simple
                activation='relu',
                solver='adam',
                max_iter=500,  # Moins d'itérations
                random_state=42,
                early_stopping=False,  # Désactivé pour éviter les problèmes
                learning_rate_init=0.001
            )

        print(f"🧠 Entraînement de l'estimateur: {estimator.__class__.__name__}...")
        estimator.fit(X_train, y_train)
        self.model = estimator
        
        # Évaluation
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Entraînement terminé")
        print(f"   Score entraînement: {train_score:.3f}")
        print(f"   Score test: {test_score:.3f}")
        
        # Prédictions détaillées
        y_pred = self.model.predict(X_test)
        print("\n📋 Rapport de classification détaillé:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Sauvegarde du modèle et du scaler
        joblib.dump(self.model, 'tempo_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        joblib.dump(self.label_encoder, 'label_encoder.joblib')
        print("💾 Modèle, scaler et encodeur sauvegardés")
        
        # Afficher l'importance des features
        self.display_feature_importance(X_train, X_test, y_test, feature_names, test_score)
        
        return True

    def log_training_performance(self, estimator_name, n_samples, accuracy, report_dict):
        """Enregistre les performances d'un entraînement dans la base de données."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        training_date = datetime.now().isoformat()
        f1_macro = report_dict.get('macro avg', {}).get('f1-score', 0)
        report_json = json.dumps(report_dict)
        
        cursor.execute('''
            INSERT INTO training_log 
            (training_date, model_algorithm, n_samples, test_accuracy, test_f1_macro, class_report_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            training_date,
            estimator_name,
            n_samples,
            accuracy,
            f1_macro,
            report_json
        ))
        
        conn.commit()
        conn.close()
        print("📝 Performances de l'entraînement enregistrées dans l'historique.")

    def display_feature_importance(self, X_train, X_test, y_test, feature_names, test_score):
        """Calcule et affiche l'importance des features en utilisant la permutation."""
        if self.model is None:
            print("⚠️  Le modèle doit être entraîné pour afficher l'importance des features.")
            return

        print("\n🔬 Calcul de l'importance des features (permutation)...")
        
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            print("❌ sklearn.inspection.permutation_importance non trouvé. Mettez à jour scikit-learn si nécessaire.")
            return

        # Utiliser f1_macro comme métrique, car c'est souvent plus pertinent pour des classes déséquilibrées
        scoring_metric = lambda estimator, X, y: f1_score(y, estimator.predict(X), average='macro')

        # Détection du debugger pour ajuster n_jobs et éviter les erreurs de multiprocessing
        n_jobs_to_use = -1
        try:
            import sys
            # La présence de 'debugpy' ou 'pydevd' dans les modules indique un debugger actif
            if 'debugpy' in sys.modules or 'pydevd' in sys.modules:
                print("⚠️  Debugger détecté — utilisation d'un seul coeur pour le calcul d'importance (n_jobs=1)")
                n_jobs_to_use = 1
        except Exception:
            # En cas d'erreur, on reste sur le comportement par défaut
            pass

        result = permutation_importance(
            self.model, X_test, y_test,
            scoring=scoring_metric,
            n_repeats=10, 
            random_state=42, 
            n_jobs=n_jobs_to_use
        )

        # Organiser les résultats
        importances = pd.Series(result.importances_mean, index=feature_names)
        importances = importances.sort_values(ascending=True)

        print("\n" + "="*50)
        print("📊 IMPORTANCE DES FEATURES")
        print("(Baisse moyenne du score F1-macro si la feature est mélangée)")
        print("="*50)
        
        for name, importance in importances.items():
            bar = "█" * int(importance * 1000) # Facteur d'échelle pour la visualisation
            print(f"{name:<20}: {importance:.4f} {bar}")
        print("="*50)

        # Enregistrement des performances
        try:
            report_dict = classification_report(y_test, self.model.predict(X_test), 
                                                  target_names=self.label_encoder.classes_,
                                                  output_dict=True)
            self.log_training_performance(self.model.__class__.__name__, len(X_train) + len(X_test), test_score, report_dict)
        except Exception as e:
            print(f"⚠️  Impossible d'enregistrer les performances de l'entraînement: {e}")
        
        return True

    def benchmark_algorithms(self, algorithms=None, cv=5, n_jobs=None):
        """Teste et compare plusieurs algorithmes sur les mêmes données.

        Args:
            algorithms (list or None): liste de clés d'algorithmes à tester. Si None, on teste un ensemble par défaut.
            cv (int): nombre de folds pour la validation croisée stratifiée.

        Retourne:
            dict: résultats résumé par algorithme (accuracy, f1 macro moyenne + std).
        """
        try:
            X, y, feature_names = self.prepare_training_data()
        except Exception as e:
            print(f"❌ Impossible de préparer les données pour le benchmark: {e}")
            return None

        # Récupérer la map d'algorithmes depuis la définition centrale
        alg_map = self.get_algorithm_map()

        if algorithms is None or len(algorithms) == 0:
            algorithms = ['mlp', 'random_forest', 'logistic', 'gb', 'svc']

        # Filtrer les clés valides
        algorithms = [a for a in algorithms if a in alg_map]
        if len(algorithms) == 0:
            print("⚠️  Aucune algorithme valide demandé pour le benchmark")
            return None

        results = {}
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Détection si un débogueur (debugpy/pydevd) est attaché qui peut casser
        # la création de sous-processus (erreur de patching args). Dans ce cas,
        # forcer n_jobs=1 pour éviter l'utilisation de multiprocessing.
        debugger_attached = False
        try:
            import sys
            # debugpy ajoute souvent 'debugpy' dans les modules importés
            if 'debugpy' in sys.modules:
                debugger_attached = True
        except Exception:
            debugger_attached = False

        if n_jobs is None:
            n_jobs_to_use = -1
        else:
            n_jobs_to_use = int(n_jobs)

        if debugger_attached and n_jobs_to_use != 1:
            print("⚠️  Debugger détecté — utilisation d'un seul coeur pour le benchmark (n_jobs=1) afin d'éviter les erreurs de subprocess)")
            n_jobs_to_use = 1

        print(f"🔬 Benchmarking {len(algorithms)} algorithmes avec {cv}-fold CV (n_jobs={n_jobs_to_use})...")
        for key in algorithms:
            clf = alg_map[key]
            print(f"  ▶️  Test de: {key} -> {clf.__class__.__name__}")
            try:
                # Use threading backend to avoid spawning subprocesses that break under some debuggers
                try:
                    from joblib import parallel_backend
                    with parallel_backend('threading', n_jobs=n_jobs_to_use):
                        scores = cross_validate(
                            clf, X, y,
                            cv=cv_split,
                            scoring=['accuracy', 'f1_macro'],
                            return_train_score=False,
                            n_jobs=n_jobs_to_use
                        )
                except Exception:
                    # Fallback to default backend if joblib.parallel_backend is unavailable
                    scores = cross_validate(
                        clf, X, y,
                        cv=cv_split,
                        scoring=['accuracy', 'f1_macro'],
                        return_train_score=False,
                        n_jobs=n_jobs_to_use
                    )

                acc_mean = scores['test_accuracy'].mean()
                acc_std = scores['test_accuracy'].std()
                f1_mean = scores['test_f1_macro'].mean()
                f1_std = scores['test_f1_macro'].std()

                results[key] = {
                    'estimator': clf.__class__.__name__,
                    'accuracy_mean': float(acc_mean),
                    'accuracy_std': float(acc_std),
                    'f1_mean': float(f1_mean),
                    'f1_std': float(f1_std),
                }

                print(f"    -> accuracy: {acc_mean:.3f} ± {acc_std:.3f}, f1_macro: {f1_mean:.3f} ± {f1_std:.3f}")
            except Exception as e:
                print(f"    ❌ Erreur lors du benchmark de {key}: {e}")
                results[key] = {'error': str(e)}

        print("\n✅ Benchmark terminé")
        # Déterminer le meilleur algorithme parmi ceux ayant réussi
        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v}
        if len(valid_results) > 0:
            # Prioriser f1_mean, puis accuracy_mean
            best_key = max(valid_results.keys(), key=lambda k: (valid_results[k].get('f1_mean', 0), valid_results[k].get('accuracy_mean', 0)))
            best = valid_results[best_key]
            results['_best'] = {
                'key': best_key,
                'estimator': best.get('estimator'),
                'f1_mean': float(best.get('f1_mean', 0)),
                'accuracy_mean': float(best.get('accuracy_mean', 0))
            }
            print(f"\n🎯 Recommandation: Utiliser '{best_key}' ({best.get('estimator')}) — f1_macro={best.get('f1_mean'):.3f}, accuracy={best.get('accuracy_mean'):.3f}")
        else:
            print("\n⚠️  Aucun algorithme n'a retourné de score valide")

        return results

    def get_algorithm_map(self):
        """Retourne un dictionnaire clé -> instance d'estimateur pour les algorithmes supportés."""
        return {
            'mlp': MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
            'svc': SVC(kernel='rbf', probability=True, random_state=42),
        }
    
    def predict_next_day(self):
        """Prédit la couleur du jour suivant avec probabilités"""
        if self.model is None:
            # Chargement du modèle sauvegardé
            try:
                self.model = joblib.load('tempo_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
                print("✅ Modèle chargé depuis les fichiers sauvegardés")
            except:
                print("❌ Aucun modèle entraîné trouvé. Veuillez d'abord entraîner le modèle.")
                return None
        
        # Récupération des données les plus récentes
        conn = sqlite3.connect(self.db_path)
        
        # Déterminer la date du jour à prédire (lendemain du dernier jour connu)
        query_date = '''
            SELECT MAX(date) FROM weather_data
        '''
        df_date = pd.read_sql_query(query_date, conn)
        last_date_str = df_date.iloc[0, 0]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        next_date = last_date + timedelta(days=1)
        next_day_of_week = next_date.weekday()  # 0=Lundi, 6=Dimanche
        
        query = '''
            SELECT 
                w.temperature_avg,
                w.temperature_max,
                w.temperature_min,
                w.precipitation,
                w.sunshine_duration,
                w.weather_code,
                w.cloud_cover,
                w.wind_speed,
                t.red_remaining,
                t.white_remaining
            FROM weather_data w
            JOIN tempo_days t ON w.date = t.date
            WHERE w.date = (SELECT MAX(date) FROM weather_data)
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print("❌ Aucune donnée récente pour la prédiction")
            return None
        
        # Préparation des features
        feature_columns = ['temperature_avg', 'temperature_max', 'temperature_min', 
                          'precipitation', 'sunshine_duration', 'weather_code', 'cloud_cover', 'wind_speed',
                          'red_remaining', 'white_remaining']
        
        # Conversion en types numériques pour les colonnes existantes
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extraire les features existantes
        features = df[feature_columns].copy()
        
        # Ajouter le jour de la semaine du jour à prédire
        features['day_of_week'] = next_day_of_week
        
        features_scaled = self.scaler.transform(features)
        
        # Prédiction des probabilités
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Création du dictionnaire de résultats
        prediction_dict = {}
        for i, prob in enumerate(probabilities):
            color_name = self.label_encoder.classes_[i]
            prediction_dict[color_name] = prob * 100
        
        # Application des règles Tempo
        day_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][next_day_of_week]
        
        if next_day_of_week == 6:  # Dimanche - toujours BLEU
            print(f"📅 Prédiction pour un {day_name} → Forcé à BLEU (règle Tempo)")
            prediction_dict = {"BLEU": 100.0, "BLANC": 0.0, "ROUGE": 0.0}
        elif next_day_of_week >= 5:  # Samedi - pas de ROUGE
            print(f"📅 Prédiction pour un {day_name} → Pas de ROUGE possible (règle Tempo)")
            rouge_prob = prediction_dict.get("ROUGE", 0)
            prediction_dict["ROUGE"] = 0.0
            # Redistribuer la probabilité rouge
            if rouge_prob > 0 and "BLEU" in prediction_dict and "BLANC" in prediction_dict:
                total_other = prediction_dict["BLEU"] + prediction_dict["BLANC"]
                if total_other > 0:
                    prediction_dict["BLEU"] += rouge_prob * (prediction_dict["BLEU"] / total_other)
                    prediction_dict["BLANC"] += rouge_prob * (prediction_dict["BLANC"] / total_other)
        else:
            print(f"📅 Prédiction pour un {day_name}")
        
        # Arrondir et normaliser à 100%
        for color in prediction_dict:
            prediction_dict[color] = round(prediction_dict[color], 1)
        
        total = sum(prediction_dict.values())
        if abs(total - 100) > 0.1:
            # Ajuste la valeur la plus élevée
            max_color = max(prediction_dict, key=prediction_dict.get)
            prediction_dict[max_color] += round(100 - total, 1)
        
        return prediction_dict
    
    def predict_multi_day(self, days=1):
        """Prédit la couleur pour les N prochains jours en simulant l'évolution du stock de jours."""
        if self.model is None:
            try:
                self.model = joblib.load('tempo_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
            except:
                print("❌ Aucun modèle entraîné trouvé.")
                return None

        conn = sqlite3.connect(self.db_path)
        
        # 1. Get current state (last known date)
        query_last = 'SELECT MAX(date) FROM weather_data'
        df_date = pd.read_sql_query(query_last, conn)
        last_date_str = df_date.iloc[0, 0]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        # Get remaining days count from the last known day
        query_state = '''
            SELECT red_remaining, white_remaining 
            FROM tempo_days 
            WHERE date = ?
        '''
        cursor = conn.cursor()
        cursor.execute(query_state, (last_date_str,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("❌ Impossible de récupérer l'état initial Tempo")
            return None
            
        current_red = row[0]
        current_white = row[1]
        
        # 2. Fetch weather forecast for N days starting from last_date + 1
        forecast_data = self.fetch_weather_forecast(days=days + 5) # Fetch a bit more to be safe
        if not forecast_data:
            return None
            
        # Filter forecast to start after last_date
        future_weather = [d for d in forecast_data if datetime.strptime(d['date'], "%Y-%m-%d") > last_date]
        future_weather = future_weather[:days]
        
        if not future_weather:
            print("❌ Pas de prévisions météo disponibles pour la période demandée")
            return None

        predictions = []
        
        for weather in future_weather:
            pred_date = datetime.strptime(weather['date'], "%Y-%m-%d")
            day_of_week = pred_date.weekday()
            
            # Prepare features
            features_dict = {
                'temperature_avg': weather['temperature_avg'],
                'temperature_max': weather['temperature_max'],
                'temperature_min': weather['temperature_min'],
                'precipitation': weather['precipitation'],
                'sunshine_duration': weather['sunshine_duration'],
                'weather_code': weather['weather_code'],
                'cloud_cover': weather['cloud_cover'],
                'wind_speed': weather['wind_speed'],
                'red_remaining': current_red,
                'white_remaining': current_white,
                'day_of_week': day_of_week
            }
            
            # Create DataFrame for scaler
            df_features = pd.DataFrame([features_dict])
            
            # Ensure column order matches training
            feature_columns = ['temperature_avg', 'temperature_max', 'temperature_min', 
                              'precipitation', 'sunshine_duration', 'weather_code', 'cloud_cover', 'wind_speed',
                              'red_remaining', 'white_remaining', 'day_of_week']
            
            df_features = df_features[feature_columns]
            
            # Scale
            features_scaled = self.scaler.transform(df_features)
            
            # Predict
            probs = self.model.predict_proba(features_scaled)[0]
            
            # Format result
            pred_dict = {"date": weather['date']}
            for i, prob in enumerate(probs):
                color = self.label_encoder.classes_[i]
                pred_dict[color] = round(prob * 100, 1)
            
            # Apply rules
            day_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week]
            if day_of_week == 6: # Dimanche
                pred_dict["BLEU"] = 100.0
                pred_dict["BLANC"] = 0.0
                pred_dict["ROUGE"] = 0.0
            elif day_of_week >= 5: # Samedi
                 rouge_prob = pred_dict.get("ROUGE", 0)
                 pred_dict["ROUGE"] = 0.0
                 if rouge_prob > 0:
                     total_other = pred_dict.get("BLEU", 0) + pred_dict.get("BLANC", 0)
                     if total_other > 0:
                         pred_dict["BLEU"] += rouge_prob * (pred_dict["BLEU"] / total_other)
                         pred_dict["BLANC"] += rouge_prob * (pred_dict["BLANC"] / total_other)

            # Normalize
            total = sum(v for k,v in pred_dict.items() if k != "date")
            if abs(total - 100) > 0.1:
                 max_color = max((k for k in pred_dict if k != "date"), key=pred_dict.get)
                 pred_dict[max_color] += round(100 - total, 1)

            predictions.append(pred_dict)
            
            # Update state for next iteration
            # Determine most likely color
            likely_color = max((k for k in pred_dict if k != "date"), key=pred_dict.get)
            
            if likely_color == "BLANC":
                current_white = max(0, current_white - 1)
            elif likely_color == "ROUGE":
                current_red = max(0, current_red - 1)
                
        return predictions

    def predict_with_custom_data(self, temperature_avg, temperature_max, temperature_min, 
                                 precipitation, sunshine_duration, weather_code, cloud_cover, wind_speed,
                                 red_remaining, white_remaining, day_of_week):
        """
        Prédit la couleur Tempo avec des données personnalisées
        
        Args:
            temperature_avg: Température moyenne (°C)
            temperature_max: Température maximale (°C)
            temperature_min: Température minimale (°C)
            precipitation: Précipitations (mm)
            sunshine_duration: Durée d'ensoleillement (heures)
            weather_code: Code météo (0-99)
            cloud_cover: Couverture nuageuse (0-100%)
            wind_speed: Vitesse du vent (km/h)
            red_remaining: Jours rouges restants (0-22)
            white_remaining: Jours blancs restants (0-43)
            day_of_week: Jour de la semaine (0=Lundi, 6=Dimanche)
        
        Returns:
            dict: Dictionnaire avec les probabilités pour chaque couleur
        """
        if self.model is None:
            # Chargement du modèle sauvegardé
            try:
                self.model = joblib.load('tempo_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
                print("✅ Modèle chargé depuis les fichiers sauvegardés")
            except:
                print("❌ Aucun modèle entraîné trouvé. Veuillez d'abord entraîner le modèle.")
                return None
        
        # Validation des entrées
        if not (0 <= red_remaining <= 22):
            print("⚠️  Avertissement: red_remaining devrait être entre 0 et 22")
        if not (0 <= white_remaining <= 43):
            print("⚠️  Avertissement: white_remaining devrait être entre 0 et 43")
        if not (0 <= cloud_cover <= 100):
            print("⚠️  Avertissement: cloud_cover devrait être entre 0 et 100")
        if not (0 <= day_of_week <= 6):
            print("⚠️  Avertissement: day_of_week devrait être entre 0 (Lundi) et 6 (Dimanche)")
        
        # Préparation des features
        features = np.array([[
            temperature_avg, temperature_max, temperature_min,
            precipitation, sunshine_duration, weather_code, cloud_cover, wind_speed,
            red_remaining, white_remaining, day_of_week
        ]])
        
        # Normalisation
        features_scaled = self.scaler.transform(features)
        
        # Prédiction
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Création du dictionnaire de résultats
        prediction_dict = {}
        for i, prob in enumerate(probabilities):
            color_name = self.label_encoder.classes_[i]
            prediction_dict[color_name] = prob * 100
        
        # Application des règles Tempo
        day_name = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week]
        
        if day_of_week == 6:  # Dimanche - toujours BLEU
            print(f"📅 Prédiction pour un {day_name} → Forcé à BLEU (règle Tempo)")
            prediction_dict = {"BLEU": 100.0, "BLANC": 0.0, "ROUGE": 0.0}
        elif day_of_week >= 5:  # Samedi - pas de ROUGE
            print(f"📅 Prédiction pour un {day_name} → Pas de ROUGE possible (règle Tempo)")
            rouge_prob = prediction_dict.get("ROUGE", 0)
            prediction_dict["ROUGE"] = 0.0
            # Redistribuer la probabilité rouge
            if rouge_prob > 0 and "BLEU" in prediction_dict and "BLANC" in prediction_dict:
                total_other = prediction_dict["BLEU"] + prediction_dict["BLANC"]
                if total_other > 0:
                    prediction_dict["BLEU"] += rouge_prob * (prediction_dict["BLEU"] / total_other)
                    prediction_dict["BLANC"] += rouge_prob * (prediction_dict["BLANC"] / total_other)
        else:
            print(f"📅 Prédiction pour un {day_name}")
        
        # Arrondir et normaliser à 100%
        for color in prediction_dict:
            prediction_dict[color] = round(prediction_dict[color], 1)
        
        total = sum(prediction_dict.values())
        if abs(total - 100) > 0.1:
            max_color = max(prediction_dict, key=prediction_dict.get)
            prediction_dict[max_color] += round(100 - total, 1)
        
        return prediction_dict
    
    def predict_sequence(self, days=9):
        """
        Prédit la séquence des couleurs pour les N prochains jours
        en mettant à jour les jours restants au fur et à mesure.
        """
        if self.model is None:
            try:
                self.model = joblib.load('tempo_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
            except:
                print("❌ Modèle non chargé")
                return None

        # 1. Récupérer l'état actuel (jours restants)
        # Essai via API externe d'abord
        current_stats = self.fetch_tempo_stats()
        
        if current_stats:
            red_remaining, white_remaining = current_stats
        else:
            # Fallback sur la base de données locale
            print("⚠️  Utilisation des données locales pour les jours restants")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT red_remaining, white_remaining FROM tempo_days ORDER BY date DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                red_remaining, white_remaining = 22, 43
            else:
                red_remaining, white_remaining = row
            

            
        # 2. Récupérer les prévisions météo
        # On demande N+1 jours pour être sûr d'avoir assez de données en excluant aujourd'hui
        forecasts = self.fetch_weather_forecast(days + 1)
        if not forecasts:
            return None
            
        # Filtrer pour ne garder que les jours à partir de demain
        today = datetime.now().strftime("%Y-%m-%d")
        future_forecasts = [f for f in forecasts if f['date'] > today]
        
        # Garder seulement le nombre de jours demandé
        future_forecasts = future_forecasts[:days]
        
        if not future_forecasts:
            print("❌ Pas assez de données futures disponibles")
            return None
            
        sequence_results = []
        
        print(f"\n🔄 Calcul de la séquence sur {len(future_forecasts)} jours (à partir de demain)...")
        print(f"   État initial : {red_remaining} rouges, {white_remaining} blancs restants")
        
        for day_weather in future_forecasts:
            date_obj = datetime.strptime(day_weather['date'], "%Y-%m-%d")
            day_of_week = date_obj.weekday()
            
            # Prédiction pour ce jour
            prediction = self.predict_with_custom_data(
                day_weather['temperature_avg'],
                day_weather['temperature_max'],
                day_weather['temperature_min'],
                day_weather['precipitation'],
                day_weather['sunshine_duration'],
                day_weather['weather_code'],
                day_weather['cloud_cover'],
                day_weather['wind_speed'],
                red_remaining,
                white_remaining,
                day_of_week
            )
            
            # Déterminer la couleur la plus probable
            most_likely_color = max(prediction, key=prediction.get)
            
            # Mettre à jour les compteurs pour le jour SUIVANT
            # Note: On décrémente seulement si on est sûr (ou on prend le plus probable)
            # Ici on simule le scénario le plus probable
            if most_likely_color == "ROUGE":
                red_remaining = max(0, red_remaining - 1)
            elif most_likely_color == "BLANC":
                white_remaining = max(0, white_remaining - 1)
                
            sequence_results.append({
                'date': day_weather['date'],
                'day_name': ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week],
                'weather_summary': f"{day_weather['temperature_avg']}°C, {day_weather['wind_speed']}km/h vent",
                'prediction': prediction,
                'chosen_color': most_likely_color,
                'remaining_after': {'red': red_remaining, 'white': white_remaining}
            })
            
        return sequence_results

    def publish_to_mqtt(self, results, broker, port, topic, user=None, password=None, discovery_prefix="homeassistant"):
        """Publie les résultats sur MQTT avec support Home Assistant Discovery"""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("❌ Erreur: La librairie 'paho-mqtt' est requise pour cette fonctionnalité.")
            print("   Installez-la avec: pip install paho-mqtt")
            return False
            
        print(f"\n📡 Connexion au broker MQTT {broker}:{port}...")
        
        client = mqtt.Client()
        
        if user and password:
            client.username_pw_set(user, password)
            
        try:
            client.connect(broker, port, 60)
            client.loop_start()
            
            # Préparation du payload global
            payload = {
                "last_update": datetime.now().isoformat(),
                "forecasts": []
            }
            
            # Device info pour HA
            device_info = {
                "identifiers": ["tempo_ia_predictor"],
                "name": "Tempo IA Predictor",
                "model": "v1.0",
                "manufacturer": "TempoIA"
            }
            
            for i, res in enumerate(results):
                day_num = i + 1
                chosen_color = res['chosen_color']
                color_emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(chosen_color, "❓")

                day_data = {
                    "date": res['date'],
                    "day_name": res['day_name'],
                    "prediction": chosen_color,
                    "prediction_emoji": color_emoji,
                    "proba_blue": res['prediction']['BLEU'],
                    "proba_white": res['prediction']['BLANC'],
                    "proba_red": res['prediction']['ROUGE'],
                    "weather_summary": res['weather_summary']
                }
                payload["forecasts"].append(day_data)
                
                # 1. Publication sur topic date (ex: tempo/forecast/2025-11-28)
                date_topic = f"{topic}/{res['date']}"
                client.publish(date_topic, json.dumps(day_data), retain=True)
                
                # 2. Publication sur topic relatif (ex: tempo/forecast/day_1)
                rel_topic = f"{topic}/day_{day_num}"
                client.publish(rel_topic, json.dumps(day_data), retain=True)
                
                # 3. Publication de la configuration pour Home Assistant Discovery
                # Un seul capteur par jour, avec les probabilités dans les attributs.
                discovery_topic = f"{discovery_prefix}/sensor/tempo_ia/day_{day_num}/config"
                discovery_payload = {
                    "name": f"Tempo J+{day_num}",
                    "unique_id": f"tempo_ia_forecast_day_{day_num}",
                    "state_topic": rel_topic,
                    "value_template": "{{ value_json.prediction_emoji }}",
                    "json_attributes_topic": rel_topic,
                    "icon": "mdi:flash",
                    "device": device_info,
                }
                client.publish(discovery_topic, json.dumps(discovery_payload), retain=True)

            
            # Publication du payload global
            client.publish(topic, json.dumps(payload), retain=True)

            print(f"✅ Données publiées sur {topic}")
            print(f"✅ {len(results)} capteurs configurés via Home Assistant Discovery")
            
            client.loop_stop()
            client.disconnect()
            return True
            
        except Exception as e:
            print(f"❌ Erreur MQTT: {e}")
            return False

    def get_database_stats(self):
        """Récupère les statistiques de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Statistiques Tempo
        cursor.execute("SELECT COUNT(*) FROM tempo_days")
        stats['total_days'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT color, COUNT(*) FROM tempo_days GROUP BY color")
        color_counts = cursor.fetchall()
        stats['color_distribution'] = {color: count for color, count in color_counts}
        
        # Statistiques météo
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        stats['total_weather_records'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                AVG(temperature_avg) as avg_temp,
                AVG(precipitation) as avg_precip,
                AVG(sunshine_duration) as avg_sunshine
            FROM weather_data
        """)
        weather_avg = cursor.fetchone()
        stats['weather_averages'] = {
            'temperature': round(weather_avg[0], 1) if weather_avg[0] else 0,
            'precipitation': round(weather_avg[1], 1) if weather_avg[1] else 0,
            'sunshine': round(weather_avg[2], 1) if weather_avg[2] else 0
        }
        
        conn.close()
        return stats
    
    def get_tempo_cycle_stats(self, year=None):
        """Retourne les statistiques détaillées d'un cycle Tempo spécifique."""
        if year is None:
            year = datetime.now().year if datetime.now().month >= 9 else datetime.now().year
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer tous les jours du cycle
        cursor.execute("""
            SELECT date, color, red_remaining, white_remaining, day_of_week
            FROM tempo_days
            WHERE cycle_year = ?
            ORDER BY date
        """, (year,))
        
        cycle_data = cursor.fetchall()
        conn.close()
        
        if not cycle_data:
            return None
        
        stats = {
            'cycle_year': year,
            'total_days': len(cycle_data),
            'colors': {'BLEU': 0, 'BLANC': 0, 'ROUGE': 0},
            'start_date': cycle_data[0][0] if cycle_data else None,
            'end_date': cycle_data[-1][0] if cycle_data else None,
            'current_remaining': {
                'red': cycle_data[-1][2] if cycle_data else 22,
                'white': cycle_data[-1][3] if cycle_data else 43
            },
            'weekday_distribution': {i: {'BLEU': 0, 'BLANC': 0, 'ROUGE': 0} for i in range(7)}
        }
        
        for date, color, red_rem, white_rem, dow in cycle_data:
            if color in stats['colors']:
                stats['colors'][color] += 1
            if dow in stats['weekday_distribution']:
                if color in stats['weekday_distribution'][dow]:
                    stats['weekday_distribution'][dow][color] += 1
        
        return stats
    
    def get_model_info(self):
        """Retourne les informations détaillées sur le modèle actuel."""
        info = {
            'model_loaded': False,
            'model_type': None,
            'training_features': [],
            'target_classes': [],
            'model_path': 'tempo_model.joblib',
            'scaler_path': 'scaler.joblib',
            'encoder_path': 'label_encoder.joblib'
        }
        
        try:
            # Charger le modèle si ce n'est pas déjà fait
            if self.model is None:
                self.model = joblib.load('tempo_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
            
            info['model_loaded'] = True
            info['model_type'] = self.model.__class__.__name__
            info['target_classes'] = self.label_encoder.classes_.tolist()
            
            # Features utilisées
            info['training_features'] = [
                'temperature_avg', 'temperature_max', 'temperature_min',
                'precipitation', 'sunshine_duration', 'weather_code',
                'cloud_cover', 'wind_speed', 'red_remaining',
                'white_remaining', 'day_of_week'
            ]
            
            # Vérifier si les fichiers existent
            info['model_file_exists'] = os.path.exists('tempo_model.joblib')
            info['scaler_file_exists'] = os.path.exists('scaler.joblib')
            info['encoder_file_exists'] = os.path.exists('label_encoder.joblib')
            
            # Obtenir la dernière entrée de training_log
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT training_date, model_algorithm, n_samples, test_accuracy, test_f1_macro
                FROM training_log
                ORDER BY training_date DESC
                LIMIT 1
            """)
            last_training = cursor.fetchone()
            conn.close()
            
            if last_training:
                info['last_training'] = {
                    'date': last_training[0],
                    'algorithm': last_training[1],
                    'n_samples': last_training[2],
                    'test_accuracy': last_training[3],
                    'test_f1_macro': last_training[4]
                }
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def get_prediction_accuracy_stats(self, limit=100):
        """
        Calcule les statistiques de précision des prédictions en comparant
        les prédictions historiques avec les couleurs réelles.
        Note: Ceci nécessiterait un système de logging des prédictions.
        Pour l'instant, retourne uniquement les métriques d'entraînement.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer l'historique des entraînements
        cursor.execute("""
            SELECT id, training_date, model_algorithm, n_samples, 
                   test_accuracy, test_f1_macro, class_report_json
            FROM training_log
            ORDER BY training_date DESC
            LIMIT ?
        """, (limit,))
        
        training_history = cursor.fetchall()
        conn.close()
        
        stats = {
            'training_history_count': len(training_history),
            'training_sessions': []
        }
        
        for row in training_history:
            session = {
                'id': row[0],
                'training_date': row[1],
                'algorithm': row[2],
                'n_samples': row[3],
                'test_accuracy': row[4],
                'test_f1_macro': row[5]
            }
            
            # Parser le rapport de classification si disponible
            if row[6]:
                try:
                    class_report = json.loads(row[6])
                    session['classification_report'] = class_report
                except:
                    pass
            
            stats['training_sessions'].append(session)
        
        # Calculer des statistiques globales
        if training_history:
            accuracies = [row[4] for row in training_history if row[4] is not None]
            f1_scores = [row[5] for row in training_history if row[5] is not None]
            
            if accuracies:
                stats['avg_accuracy'] = sum(accuracies) / len(accuracies)
                stats['max_accuracy'] = max(accuracies)
                stats['min_accuracy'] = min(accuracies)
            
            if f1_scores:
                stats['avg_f1_macro'] = sum(f1_scores) / len(f1_scores)
                stats['max_f1_macro'] = max(f1_scores)
                stats['min_f1_macro'] = min(f1_scores)
        
        return stats
    
    def initialize_database(self, years=3):
        """Initialise ou met à jour la base de données avec plusieurs années de données."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                print("🗑️  Ancienne base de données supprimée")
            except OSError as e:
                print(f"⚠️  Impossible de supprimer la base: {e}")

        self.create_database()

        today = datetime.now()
        current_cycle_year = today.year if today.month >= 9 else today.year - 1

        # Déterminer l'année de cycle la plus récente à récupérer
        # Si nous sommes après le 1er septembre, le cycle le plus récent est l'année en cours + 1
        # (ex: en oct 2024, le cycle est 2024-2025, donc l'année de fin est 2025)
        latest_cycle_end_year = today.year + 1 if today.month >= 9 else today.year

        for i in range(years):
            year = latest_cycle_end_year - i
            print(f"\n📅 Traitement du cycle se terminant en {year}...")

            start_date_str = f"{year-1}-09-01"
            end_date_str = f"{year}-08-31"

            # Si la date de fin du cycle est dans le futur, la plafonner à aujourd'hui
            end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d")
            if end_date_obj > today:
                end_date_str = today.strftime("%Y-%m-%d")
                print(f"   (Cycle en cours, données jusqu'au {end_date_str})")

            tempo_data = self.fetch_tempo_data(year)
            tempo_data_with_remaining = self.calculate_remaining_days(tempo_data, year)
            self.insert_tempo_data(tempo_data_with_remaining)

            weather_data = self.fetch_historical_weather(start_date_str, end_date_str)
            self.insert_weather_data(weather_data)
        
        print("\n✅ Base de données initialisée avec succès!")

    def update_current_cycle(self):
        """Met à jour le cycle en cours avec les dernières données sans tout supprimer."""
        print("🔄 Mise à jour du cycle en cours...")

        # 1. Déterminer le cycle en cours
        today = datetime.now()
        cycle_end_year = today.year + 1 if today.month >= 9 else today.year
        
        # 2. Trouver la dernière date dans la base de données
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM tempo_days")
        last_date_in_db_str = cursor.fetchone()[0]
        conn.close()

        if not last_date_in_db_str:
            print("⚠️  Base de données vide. Lancement d'une initialisation complète.")
            self.initialize_database(years=7)
            return

        last_date_in_db = datetime.strptime(last_date_in_db_str, "%Y-%m-%d")
        start_date_obj = last_date_in_db + timedelta(days=1)

        # Si la base est déjà à jour, on arrête
        if start_date_obj.date() > today.date():
            print("✅ La base de données est déjà à jour.")
            return False

        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = today.strftime("%Y-%m-%d")

        print(f"📅 Récupération des données du {start_date_str} au {end_date_str}...")

        # 3. Récupérer et insérer les données Tempo
        # fetch_tempo_data récupère tout le cycle, mais insert_tempo_data avec OR REPLACE
        # mettra à jour efficacement les jours existants et ajoutera les nouveaux.
        tempo_data = self.fetch_tempo_data(cycle_end_year)
        tempo_data_with_remaining = self.calculate_remaining_days(tempo_data, cycle_end_year)
        self.insert_tempo_data(tempo_data_with_remaining)

        # 4. Récupérer et insérer les données météo pour la période manquante
        weather_data = self.fetch_historical_weather(start_date_str, end_date_str)
        self.insert_weather_data(weather_data)

        print("\n✅ Mise à jour du cycle en cours terminée avec succès!")
        return True

class DatabaseVisualizer:
    """Classe pour visualiser et exporter les données de la base Tempo"""
    
    def __init__(self, db_path='tempo_weather.db'):
        self.db_path = db_path
    
    def view_tempo_data(self, limit=20):
        """Affiche les données Tempo"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT date, color, red_remaining, white_remaining, cycle_year
            FROM tempo_days
            ORDER BY date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if len(df) == 0:
            print("❌ Aucune donnée Tempo dans la base")
            return
        
        print(f"\n📊 DONNÉES TEMPO (derniers {limit} jours)")
        print("=" * 80)
        print(f"{'Date':<12} {'Couleur':<10} {'Rouge restant':<15} {'Blanc restant':<15} {'Cycle':<6}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            color_emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(row['color'], "❓")
            print(f"{row['date']:<12} {color_emoji} {row['color']:<8} {row['red_remaining']:<15} "
                  f"{row['white_remaining']:<15} {row['cycle_year']:<6}")
        
        print("-" * 80)
        
        # Statistiques de distribution
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT color, COUNT(*) FROM tempo_days GROUP BY color")
        color_stats = cursor.fetchall()
        conn.close()
        
        print("\n📈 DISTRIBUTION DES COULEURS:")
        for color, count in color_stats:
            color_emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(color, "❓")
            bar = "█" * (count // 10)
            print(f"  {color_emoji} {color:<6}: {count:>4} jours {bar}")
    
    def view_weather_data(self, limit=20):
        """Affiche les données météo"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT date, temperature_avg, temperature_max, temperature_min,
                   precipitation, sunshine_duration, weather_code, cloud_cover, wind_speed
            FROM weather_data
            ORDER BY date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if len(df) == 0:
            print("❌ Aucune donnée météo dans la base")
            return
        
        # Conversion des colonnes en types numériques
        numeric_cols = ['temperature_avg', 'temperature_max', 'temperature_min', 
                       'precipitation', 'sunshine_duration', 'weather_code', 'cloud_cover', 'wind_speed']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"\n🌤️  DONNÉES MÉTÉO (derniers {limit} jours)")
        print("=" * 120)
        print(f"{'Date':<12} {'T.Moy':<8} {'T.Max':<8} {'T.Min':<8} "
              f"{'Précip':<10} {'Soleil':<10} {'Nuages':<10} {'Vent':<8} {'Code':<6}")
        print("-" * 120)
        
        for _, row in df.iterrows():
            print(f"{row['date']:<12} {row['temperature_avg']:>6.1f}°C "
                  f"{row['temperature_max']:>6.1f}°C {row['temperature_min']:>6.1f}°C "
                  f"{row['precipitation']:>8.1f}mm {row['sunshine_duration']:>8.1f}h "
                  f"{row['cloud_cover']:>8.0f}% {row['wind_speed']:>6.1f} {row['weather_code']:>6.0f}")
        
        print("-" * 120)
        
        # Statistiques moyennes
        print("\n📊 MOYENNES SUR TOUTES LES DONNÉES:")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                AVG(temperature_avg), AVG(temperature_max), AVG(temperature_min),
                AVG(precipitation), AVG(sunshine_duration), AVG(cloud_cover), AVG(wind_speed)
            FROM weather_data
        """)
        avgs = cursor.fetchone()
        conn.close()
        
        print(f"  🌡️  Température moyenne: {avgs[0]:.1f}°C")
        print(f"  🌡️  Température max moy: {avgs[1]:.1f}°C")
        print(f"  🌡️  Température min moy: {avgs[2]:.1f}°C")
        print(f"  💧 Précipitations moy: {avgs[3]:.1f}mm")
        print(f"  ☀️  Ensoleillement moy: {avgs[4]:.1f}h")
        print(f"  ☁️  Couverture nuageuse moy: {avgs[5]:.0f}%")
        print(f"  💨 Vitesse vent moy: {avgs[6]:.1f} km/h")
    
    def view_combined_data(self, limit=10):
        """Affiche les données combinées Tempo + Météo"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                t.date, t.color, t.red_remaining, t.white_remaining,
                w.temperature_avg, w.precipitation, w.sunshine_duration
            FROM tempo_days t
            JOIN weather_data w ON t.date = w.date
            ORDER BY t.date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if len(df) == 0:
            print("❌ Aucune donnée combinée dans la base")
            return
        
        print(f"\n🔗 DONNÉES COMBINÉES TEMPO + MÉTÉO (derniers {limit} jours)")
        print("=" * 110)
        print(f"{'Date':<12} {'Couleur':<10} {'R.Res':<6} {'B.Res':<6} "
              f"{'T.Moy':<8} {'Précip':<10} {'Soleil':<10}")
        print("-" * 110)
        
        for _, row in df.iterrows():
            color_emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(row['color'], "❓")
            print(f"{row['date']:<12} {color_emoji} {row['color']:<8} "
                  f"{row['red_remaining']:<6} {row['white_remaining']:<6} "
                  f"{row['temperature_avg']:>6.1f}°C {row['precipitation']:>8.1f}mm "
                  f"{row['sunshine_duration']:>8.1f}h")
        
        print("-" * 110)
    
    def display_statistics(self):
        """Affiche des statistiques globales"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n📊 STATISTIQUES GLOBALES")
        print("=" * 80)
        
        # Nombre total d'enregistrements
        cursor.execute("SELECT COUNT(*) FROM tempo_days")
        total_tempo = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        total_weather = cursor.fetchone()[0]
        
        print(f"\n📅 Enregistrements:")
        print(f"  - Jours Tempo: {total_tempo}")
        print(f"  - Jours météo: {total_weather}")
        
        # Distribution des couleurs
        cursor.execute("SELECT color, COUNT(*) FROM tempo_days GROUP BY color")
        color_dist = cursor.fetchall()
        
        print(f"\n🎨 Distribution des couleurs:")
        total_colored = sum(count for _, count in color_dist)
        for color, count in color_dist:
            percentage = (count / total_colored * 100) if total_colored > 0 else 0
            color_emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(color, "❓")
            bar = "█" * int(percentage / 2)
            print(f"  {color_emoji} {color:<6}: {count:>4} ({percentage:>5.1f}%) {bar}")
        
        # Plages de températures
        cursor.execute("""
            SELECT MIN(temperature_min), MAX(temperature_max), AVG(temperature_avg)
            FROM weather_data
        """)
        temp_stats = cursor.fetchone()
        
        print(f"\n🌡️  Températures:")
        print(f"  - Minimum absolu: {temp_stats[0]:.1f}°C")
        print(f"  - Maximum absolu: {temp_stats[1]:.1f}°C")
        print(f"  - Moyenne globale: {temp_stats[2]:.1f}°C")
        
        # Cycles disponibles
        cursor.execute("SELECT DISTINCT cycle_year FROM tempo_days ORDER BY cycle_year")
        cycles = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📆 Cycles disponibles: {', '.join(map(str, cycles))}")
        
        conn.close()
        print("=" * 80)
    
    def export_to_csv(self, output_dir='.'):
        """Exporte les données en fichiers CSV"""
        conn = sqlite3.connect(self.db_path)
        
        # Export des données Tempo
        tempo_df = pd.read_sql_query("SELECT * FROM tempo_days ORDER BY date", conn)
        tempo_file = f"{output_dir}/tempo_data.csv"
        tempo_df.to_csv(tempo_file, index=False, encoding='utf-8')
        print(f"✅ Données Tempo exportées: {tempo_file} ({len(tempo_df)} lignes)")
        
        # Export des données météo
        weather_df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY date", conn)
        weather_file = f"{output_dir}/weather_data.csv"
        weather_df.to_csv(weather_file, index=False, encoding='utf-8')
        print(f"✅ Données météo exportées: {weather_file} ({len(weather_df)} lignes)")
        
        # Export des données combinées
        combined_query = '''
            SELECT 
                t.date, t.color, t.red_remaining, t.white_remaining, t.cycle_year,
                w.temperature_avg, w.temperature_max, w.temperature_min,
                w.precipitation, w.sunshine_duration, w.weather_code
            FROM tempo_days t
            JOIN weather_data w ON t.date = w.date
            ORDER BY t.date
        '''
        combined_df = pd.read_sql_query(combined_query, conn)
        combined_file = f"{output_dir}/combined_data.csv"
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"✅ Données combinées exportées: {combined_file} ({len(combined_df)} lignes)")
        
        conn.close()
        print(f"\n📁 Tous les fichiers exportés dans: {output_dir}/")

    def view_performance_history(self, limit=20, return_df=False):
        """Affiche l'historique des performances d'entraînement."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT id, training_date, model_algorithm, n_samples, test_accuracy, test_f1_macro
            FROM training_log
            ORDER BY training_date DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if return_df:
            return df if len(df) > 0 else None

        if len(df) == 0:
            print("❌ Aucun historique de performance trouvé. Entraînez d'abord un modèle avec --train.")
            return

        print(f"\n📜 HISTORIQUE DES PERFORMANCES (derniers {limit} entraînements)")
        print("=" * 110)
        print(f"{'Date':<26} {'Algorithme':<25} {'Échantillons':<12} {'Précision':<22} {'Score F1 (macro)':<22}")
        print("-" * 110)

        for _, row in df.iterrows():
            date = datetime.fromisoformat(row['training_date']).strftime('%Y-%m-%d %H:%M:%S')
            accuracy = row['test_accuracy']
            f1_macro = row['test_f1_macro']
            
            acc_bar = "█" * int(accuracy * 20)
            f1_bar = "█" * int(f1_macro * 20)

            print(f"{date:<26} {row['model_algorithm']:<25} {row['n_samples']:<12} {accuracy:.3f} {acc_bar:<20} {f1_macro:.3f} {f1_bar:<20}")
        
        print("=" * 110)

    def get_training_log_details(self, training_id: int):
        """Récupère les détails d'un enregistrement d'entraînement spécifique par son ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = 'SELECT id, training_date, model_algorithm, n_samples, test_accuracy, test_f1_macro, class_report_json FROM training_log WHERE id = ?'
        cursor.execute(query, (training_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        report_dict = json.loads(row[6])
        return {"id": row[0], "training_date": row[1], "model_algorithm": row[2], "n_samples": row[3], "test_accuracy": row[4], "test_f1_macro": row[5], "classification_report": report_dict}

def main():
    parser = argparse.ArgumentParser(
        description='Prédiction des couleurs Tempo EDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python tempoia.py --init-db              # Initialise la base de données
  python tempoia.py --train                # Entraîne le modèle
  python tempoia.py --predict              # Prédit le jour suivant
  python tempoia.py --view-db              # Visualise la base de données
  python tempoia.py --export-csv           # Exporte les données en CSV
  python tempoia.py --predict-custom       # Prédiction avec données personnalisées
  python tempoia.py --interactive          # Mode interactif
        """
    )
    
    # Options existantes
    parser.add_argument('--init-db', action='store_true', 
                       help='Initialiser la base de données')
    parser.add_argument('--update', action='store_true',
                       help='Mettre à jour uniquement le cycle en cours avec les dernières données')
    parser.add_argument('--train', action='store_true', 
                       help='Entraîner le modèle de prédiction')
    parser.add_argument('--predict', action='store_true', 
                       help='Prédire la couleur du jour suivant')
    parser.add_argument('--years', type=int, default=3, 
                       help='Nombre d\'années de données à charger (défaut: 3)')
    
    # Nouvelles options de visualisation
    parser.add_argument('--view-db', action='store_true',
                       help='Visualiser les données de la base')
    parser.add_argument('--view-tempo', action='store_true',
                       help='Afficher uniquement les données Tempo')
    parser.add_argument('--view-weather', action='store_true',
                       help='Afficher uniquement les données météo')
    parser.add_argument('--view-combined', action='store_true',
                       help='Afficher les données combinées')
    parser.add_argument('--stats', action='store_true',
                       help='Afficher les statistiques globales')
    parser.add_argument('--feature-importance', action='store_true',
                       help="Calcule et affiche l'importance des features du modèle entraîné")
    parser.add_argument('--performance', action='store_true',
                       help="Affiche l'historique des performances du modèle")
    parser.add_argument('--limit', type=int, default=20,
                       help='Nombre de lignes à afficher (défaut: 20)')
    
    # Options d'export
    parser.add_argument('--export-csv', action='store_true',
                       help='Exporter les données en CSV')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Répertoire de sortie pour les exports (défaut: .)')
    
    # Option de prédiction sur 9 jours (ou N jours)
    parser.add_argument('--forecast', nargs='?', const=9, type=int,
                       help='Prédire la séquence sur N jours (défaut: 9)')
    
    # Option de prédiction personnalisée
    parser.add_argument('--predict-custom', action='store_true',
                       help='Faire une prédiction avec saisie manuelle')
                       
    # Options MQTT / Mode Auto
    parser.add_argument('--auto-mqtt', action='store_true',
                       help='Mode auto: Init DB + Train + Forecast 14j + MQTT')
    parser.add_argument('--mqtt-broker', type=str, default='localhost',
                       help='Adresse du broker MQTT')
    parser.add_argument('--auto-benchmark', action='store_true',
                       help='En mode auto, lance un benchmark et utilise le meilleur algo pour l\'entraînement')
    parser.add_argument('--mqtt-port', type=int, default=1883,
                       help='Port du broker MQTT')
    parser.add_argument('--mqtt-topic', type=str, default='tempo/forecast',
                       help='Topic MQTT racine')
    parser.add_argument('--mqtt-discovery-prefix', type=str, default='homeassistant',
                       help='Préfixe pour Home Assistant Discovery (défaut: homeassistant)')
    parser.add_argument('--mqtt-user', type=str, help='Utilisateur MQTT')
    parser.add_argument('--mqtt-password', type=str, help='Mot de passe MQTT')
    
    # Mode interactif
    parser.add_argument('--interactive', action='store_true',
                       help='Lancer le mode interactif')

    # Benchmarking algorithms
    parser.add_argument('--benchmark-algos', action='store_true',
                       help='Benchmark/tester plusieurs algorithmes ML (cross-validation)')
    parser.add_argument('--algos', nargs='*',
                       help='Liste d\'algorithmes à tester: mlp, random_forest, logistic, gb, svc')
    parser.add_argument('--cv', type=int, default=5,
                       help='Nombre de folds pour la validation croisée (défaut: 5)')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Nombre de jobs pour cross_validate (par défaut -1). Forcé à 1 si un debugger est détecté.')

    # Option pour forcer/choisir l'algorithme
    parser.add_argument('--algorithm', type=str,
                       help="Forcer l'algorithme à utiliser pour l'entraînement: mlp, random_forest, logistic, gb, svc")
    parser.add_argument('--select-algo', action='store_true',
                       help='Lancer le benchmark puis entraîner automatiquement le meilleur algorithme trouvé')
    
    args = parser.parse_args()
    
    predictor = TempoWeatherPredictor()
    visualizer = DatabaseVisualizer()
    
    # Mode interactif
    if args.interactive:
        interactive_mode(predictor, visualizer)
        return

    # Benchmark des algorithmes
    if args.benchmark_algos:
        print("🔬 Démarrage du benchmark des algorithmes...")
        results = predictor.benchmark_algorithms(algorithms=args.algos, cv=args.cv, n_jobs=args.n_jobs)
        if results:
            try:
                import json
                print("\n📊 Résultats du benchmark:")
                print(json.dumps(results, indent=2, ensure_ascii=False))
            except Exception:
                print(results)
        return
    
    # Mode Automatique MQTT
    if args.auto_mqtt:
        print("🚀 Démarrage du mode automatique MQTT")
        
        # 1. Mise à jour de la base de données
        print("1️⃣  Mise à jour de la base de données...")
        updated = predictor.update_current_cycle()

        # 2. Entraînement conditionnel (uniquement si de nouvelles données sont disponibles)
        if updated:
            print("\n2️⃣  Entraînement du modèle car de nouvelles données sont disponibles...")
            estimator_to_use = args.algorithm
            if args.auto_benchmark:
                print("🔬 Lancement du benchmark pour sélectionner le meilleur algorithme...")
                benchmark_results = predictor.benchmark_algorithms()
                if benchmark_results and '_best' in benchmark_results:
                    estimator_to_use = benchmark_results['_best']['key']
                    print(f"🎯 Meilleur algorithme trouvé: {estimator_to_use}")
                else:
                    print("⚠️  Benchmark échoué, utilisation de l'algorithme par défaut.")
            predictor.train_model(estimator_key=estimator_to_use)
        else:
            print("\n2️⃣  Base de données déjà à jour. Saut de l'entraînement.")

        # 3. Prévisions (toujours exécutées)
        print("\n3️⃣  Prévisions sur 14 jours...")
        results = predictor.predict_sequence(days=14)
        
        # 4. Envoi MQTT (toujours exécuté si les prévisions ont réussi)
        if results:
            print("\n4️⃣  Envoi vers MQTT...")
            predictor.publish_to_mqtt(
                results, 
                args.mqtt_broker, 
                args.mqtt_port, 
                args.mqtt_topic,
                args.mqtt_user, 
                args.mqtt_password,
                args.mqtt_discovery_prefix
            )
        else:
            print("❌ Échec des prévisions, pas d'envoi MQTT.")
        return

    # Initialisation de la base de données
    if args.init_db:
        print("🚀 Initialisation de la base de données...")
        predictor.initialize_database(args.years)
    
    # Mise à jour du cycle en cours
    if args.update:
        print("🚀 Mise à jour du cycle en cours...")
        predictor.update_current_cycle()

    # Entraînement du modèle
    if args.train:
        success = predictor.train_model(estimator_key=args.algorithm)
        if not success:
            print("❌ Échec de l'entraînement. Vérifiez que la base de données contient des données.")
    
    # Prédiction standard
    if args.predict:
        print("🔮 Prédiction de la couleur du jour suivant...")
        prediction = predictor.predict_next_day()
        
        if prediction:
            print("\n🎯 PRÉDICTION TEMPO:")
            print(json.dumps(prediction, indent=2, ensure_ascii=False))
            
            # Vérification que la somme fait bien 100%
            total = sum(prediction.values())
            print(f"\n✅ Somme des probabilités: {total}%")
            
            # Affichage de la couleur la plus probable
            most_likely = max(prediction, key=prediction.get)
            emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(most_likely, "❓")
            print(f"\n🎯 Couleur la plus probable: {emoji} {most_likely} ({prediction[most_likely]}%)")
            
    # Prédiction séquence N jours
    if args.forecast:
        days = args.forecast
        results = predictor.predict_sequence(days)
        if results:
            print(f"\n🔮 PRÉVISIONS SUR {days} JOURS")
            print("=" * 100)
            print(f"{'Date':<12} {'Jour':<10} {'Météo':<20} {'Prédiction':<30} {'Reste (R/B)'}")
            print("-" * 100)
            
            for res in results:
                pred_str = f"{res['chosen_color']} ({res['prediction'][res['chosen_color']]:.0f}%)"
                emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(res['chosen_color'], "")
                rem_str = f"{res['remaining_after']['red']}/{res['remaining_after']['white']}"
                
                print(f"{res['date']:<12} {res['day_name']:<10} {res['weather_summary']:<20} {emoji} {pred_str:<27} {rem_str}")
            print("=" * 100)
    
    # Prédiction personnalisée
    if args.predict_custom:
        custom_prediction_interactive(predictor)
    
    # Visualisation de la base de données
    if args.view_db or args.view_tempo or args.view_weather or args.view_combined or args.stats:
        if args.view_tempo or args.view_db:
            visualizer.view_tempo_data(args.limit)
        
        if args.view_weather or args.view_db:
            visualizer.view_weather_data(args.limit)
        
        if args.view_combined or args.view_db:
            visualizer.view_combined_data(min(args.limit, 10))
        
        if args.stats or args.view_db:
            visualizer.display_statistics()
    
    # Export CSV
    if args.export_csv:
        print("\n📤 Export des données en CSV...")
        visualizer.export_to_csv(args.output_dir)
    
    # Affichage de l'aide si aucune option
    if args.select_algo:
        print("🔬 Exécution du benchmark pour sélectionner le meilleur algorithme...")
        results = predictor.benchmark_algorithms(algorithms=args.algos, cv=args.cv, n_jobs=args.n_jobs)
        if results and '_best' in results:
            best_key = results['_best']['key']
            print(f"🧠 Entraînement automatique avec l'algorithme recommandé: {best_key}")
            success = predictor.train_model(estimator_key=best_key)
            if not success:
                print("❌ Échec de l'entraînement automatique du meilleur algorithme.")
        else:
            print("❌ Aucun algorithme valide trouvé durant le benchmark. Annulation de l'entraînement automatique.")
        return

    # Affichage de l'importance des features
    if args.feature_importance:
        print("🔬 Calcul de l'importance des features...")
        try:
            features, target, feature_names = predictor.prepare_training_data()
            _, X_test, _, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            predictor.model = joblib.load('tempo_model.joblib')
            predictor.display_feature_importance(X_train, X_test, y_test, feature_names, predictor.model.score(X_test, y_test))
        except FileNotFoundError:
            print("❌ Le modèle 'tempo_model.joblib' n'a pas été trouvé. Veuillez d'abord entraîner le modèle avec --train.")
        except ValueError as e:
            print(f"❌ Erreur lors de la préparation des données: {e}")
        except Exception as e:
            print(f"❌ Une erreur est survenue: {e}")
        return

    # Affichage de l'historique des performances
    if args.performance:
        visualizer.view_performance_history()
        return

    if not any([args.init_db, args.update, args.train, args.predict, args.view_db, 
                args.view_tempo, args.view_weather, args.view_combined, args.stats,
                args.export_csv, args.predict_custom, args.interactive, args.forecast,
                args.performance, args.feature_importance, args.benchmark_algos, args.select_algo]):
        print("ℹ️  Utilisez --help pour voir les options disponibles")
        print("💡 Ou utilisez --interactive pour le mode interactif")

def custom_prediction_interactive(predictor):
    """Mode interactif pour la prédiction personnalisée"""
    print("\n🔮 PRÉDICTION PERSONNALISÉE")
    print("=" * 80)
    print("Entrez les paramètres météo et les jours restants pour faire une prédiction.")
    print("(Appuyez sur Entrée pour utiliser les valeurs par défaut)")
    print()
    
    try:
        # Récupération des valeurs par défaut depuis la base
        conn = sqlite3.connect('tempo_weather.db')
        cursor = conn.cursor()
        
        # Dernières valeurs connues
        cursor.execute("""
            SELECT t.red_remaining, t.white_remaining,
                   w.temperature_avg, w.temperature_max, w.temperature_min,
                   w.precipitation, w.sunshine_duration, w.weather_code, w.cloud_cover, w.wind_speed,
                   w.date
            FROM tempo_days t
            JOIN weather_data w ON t.date = w.date
            ORDER BY t.date DESC
            LIMIT 1
        """)
        defaults = cursor.fetchone()
        conn.close()
        
        if defaults:
            def_red, def_white, def_tavg, def_tmax, def_tmin, def_precip, def_sun, def_code, def_cloud, def_wind, last_date = defaults
            # Calculer le jour suivant pour la prédiction
            last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
            next_date = last_date_obj + timedelta(days=1)
            def_day_of_week = next_date.weekday()
        else:
            def_red, def_white = 15, 30
            def_tavg, def_tmax, def_tmin = 10.0, 15.0, 5.0
            def_precip, def_sun, def_code, def_cloud, def_wind = 2.0, 5.0, 1, 50.0, 15.0
            def_day_of_week = 0  # Lundi
            next_date = datetime.now() + timedelta(days=1)
        
        # Afficher la date suggérée
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        print(f"📅 Date suggérée pour la prédiction : {next_date.strftime('%Y-%m-%d')} ({day_names[def_day_of_week]})")
        print()
        
        # Saisie des paramètres
        temp_avg = input(f"🌡️  Température moyenne (°C) [{def_tavg:.1f}]: ") or str(def_tavg)
        temp_max = input(f"🌡️  Température maximale (°C) [{def_tmax:.1f}]: ") or str(def_tmax)
        temp_min = input(f"🌡️  Température minimale (°C) [{def_tmin:.1f}]: ") or str(def_tmin)
        precipitation = input(f"💧 Précipitations (mm) [{def_precip:.1f}]: ") or str(def_precip)
        sunshine = input(f"☀️  Ensoleillement (heures) [{def_sun:.1f}]: ") or str(def_sun)
        weather_code = input(f"🌤️  Code météo (0-99) [{def_code}]: ") or str(def_code)
        cloud_cover = input(f"☁️  Couverture nuageuse (%) [{def_cloud:.0f}]: ") or str(def_cloud)
        wind_speed = input(f"💨 Vitesse du vent (km/h) [{def_wind:.1f}]: ") or str(def_wind)
        red_remaining = input(f"🔴 Jours rouges restants (0-22) [{def_red}]: ") or str(def_red)
        white_remaining = input(f"⚪ Jours blancs restants (0-43) [{def_white}]: ") or str(def_white)
        
        # Demander le jour de la semaine
        print(f"\n📅 Jour de la semaine (0=Lundi, 1=Mardi, ..., 6=Dimanche)")
        day_of_week_input = input(f"   [{def_day_of_week} - {day_names[def_day_of_week]}]: ") or str(def_day_of_week)
        
        # Conversion en nombres
        temp_avg = float(temp_avg)
        temp_max = float(temp_max)
        temp_min = float(temp_min)
        precipitation = float(precipitation)
        sunshine = float(sunshine)
        weather_code = int(weather_code)
        cloud_cover = float(cloud_cover)
        wind_speed = float(wind_speed)
        red_remaining = int(red_remaining)
        white_remaining = int(white_remaining)
        day_of_week = int(day_of_week_input)
        
        print("\n🔄 Prédiction en cours...")
        
        # Prédiction
        prediction = predictor.predict_with_custom_data(
            temp_avg, temp_max, temp_min,
            precipitation, sunshine, weather_code, cloud_cover, wind_speed,
            red_remaining, white_remaining, day_of_week
        )
        
        if prediction:
            print("\n" + "=" * 80)
            print("🎯 RÉSULTAT DE LA PRÉDICTION")
            print("=" * 80)
            
            # Affichage des probabilités
            for color in ["BLEU", "BLANC", "ROUGE"]:
                if color in prediction:
                    emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}[color]
                    prob = prediction[color]
                    bar = "█" * int(prob / 2)
                    print(f"{emoji} {color:<6}: {prob:>5.1f}% {bar}")
            
            # Couleur la plus probable
            most_likely = max(prediction, key=prediction.get)
            emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(most_likely, "❓")
            print(f"\n🎯 Couleur la plus probable: {emoji} {most_likely} ({prediction[most_likely]}%)")
            print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n❌ Annulé par l'utilisateur")
    except ValueError as e:
        print(f"\n❌ Erreur de saisie: {e}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

def interactive_mode(predictor, visualizer):
    """Mode interactif avec menu"""
    while True:
        print("\n" + "=" * 80)
        print("🎯 PRÉDICTEUR TEMPO EDF - MODE INTERACTIF")
        print("=" * 80)
        print("\n📋 MENU PRINCIPAL:")
        print("  1. 📊 Visualiser la base de données")
        print("  2. 🔮 Faire une prédiction")
        print("  3. 🧠 Entraîner le modèle")
        print("  4. 🗄️  Gestion de la base de données")
        print("  5. 📤 Exporter les données en CSV")
        print("  6. 📈 Afficher les statistiques")
        print("  7. 🔬 Analyser le modèle (Importance des features)")
        print("  8. 📜 Afficher l'historique des performances")
        print("  9. ⚙️  Benchmark des algorithmes")
        print("  10. ❌ Quitter")
        print()
        
        try:
            choice = input("Votre choix (1-10): ").strip()
            
            if choice == "1":
                # Sous-menu visualisation
                print("\n📊 VISUALISATION:")
                print("  1. Données Tempo")
                print("  2. Données météo")
                print("  3. Données combinées")
                print("  4. Tout afficher")
                
                sub_choice = input("\nVotre choix (1-4): ").strip()
                limit = input("Nombre de lignes (défaut 20): ").strip() or "20"
                limit = int(limit)
                
                if sub_choice == "1":
                    visualizer.view_tempo_data(limit)
                elif sub_choice == "2":
                    visualizer.view_weather_data(limit)
                elif sub_choice == "3":
                    visualizer.view_combined_data(min(limit, 10))
                elif sub_choice == "4":
                    visualizer.view_tempo_data(limit)
                    visualizer.view_weather_data(limit)
                    visualizer.view_combined_data(min(limit, 10))
            
            elif choice == "2":
                # Sous-menu prédiction
                print("\n🔮 PRÉDICTION:")
                print("  1. Prédire le jour suivant (données automatiques)")
                print("  2. Prévisions sur 9 jours (Séquence)")
                print("  3. Prédiction personnalisée")
                
                sub_choice = input("\nVotre choix (1-3): ").strip()
                
                if sub_choice == "1":
                    prediction = predictor.predict_next_day()
                    if prediction:
                        print("\n🎯 PRÉDICTION:")
                        for color, prob in prediction.items():
                            emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(color, "❓")
                            bar = "█" * int(prob / 2)
                            print(f"{emoji} {color:<6}: {prob:>5.1f}% {bar}")
                        
                        most_likely = max(prediction, key=prediction.get)
                        emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(most_likely, "❓")
                        print(f"\n🎯 Couleur la plus probable: {emoji} {most_likely}")
                
                elif sub_choice == "2":
                    try:
                        days_input = input("\nCombien de jours voulez-vous prédire ? [9]: ").strip()
                        days = int(days_input) if days_input else 9
                    except ValueError:
                        print("⚠️  Valeur invalide, utilisation de 9 jours par défaut.")
                        days = 9
                        
                    results = predictor.predict_sequence(days)
                    if results:
                        print(f"\n🔮 PRÉVISIONS SUR {days} JOURS")
                        print("=" * 100)
                        print(f"{'Date':<12} {'Jour':<10} {'Météo':<20} {'Prédiction':<30} {'Reste (R/B)'}")
                        print("-" * 100)
                        
                        for res in results:
                            pred_str = f"{res['chosen_color']} ({res['prediction'][res['chosen_color']]:.0f}%)"
                            emoji = {"BLEU": "🔵", "BLANC": "⚪", "ROUGE": "🔴"}.get(res['chosen_color'], "")
                            rem_str = f"{res['remaining_after']['red']}/{res['remaining_after']['white']}"
                            
                            print(f"{res['date']:<12} {res['day_name']:<10} {res['weather_summary']:<20} {emoji} {pred_str:<27} {rem_str}")
                        print("=" * 100)
                
                elif sub_choice == "3":
                    custom_prediction_interactive(predictor)

            elif choice == "3":
                print("\n🧠 Entraînement du modèle...")
                print("Choisissez un algorithme à forcer (mlp, random_forest, logistic, gb, svc)")
                print("Laissez vide pour utiliser le comportement par défaut (MLP). Tapez 'best' pour lancer un benchmark et entraîner automatiquement le meilleur.")
                algo_choice = input("Algorithme (mlp|random_forest|logistic|gb|svc|'best'|Entrée): ").strip()

                if algo_choice.lower() == 'best':
                    print("🔬 Lancement du benchmark pour sélectionner le meilleur algorithme...")
                    results = predictor.benchmark_algorithms(algorithms=None, cv=5, n_jobs=None)
                    if results and '_best' in results:
                        best_key = results['_best']['key']
                        print(f"🧠 Entraînement automatique avec l'algorithme recommandé: {best_key}")
                        predictor.train_model(estimator_key=best_key)
                    else:
                        print("❌ Aucun algorithme valide trouvé durant le benchmark. Annulation.")
                else:
                    estimator_key = algo_choice if algo_choice != '' else None
                    # Validate if estimator_key provided
                    if estimator_key:
                        alg_map = predictor.get_algorithm_map()
                        if estimator_key not in alg_map:
                            print(f"❌ Algorithme inconnu: {estimator_key}. Clés valides: {list(alg_map.keys())}")
                        else:
                            predictor.train_model(estimator_key=estimator_key)
                    else:
                        predictor.train_model()
            
            elif choice == "4":
                # Sous-menu Gestion de la base de données
                print("\n🗄️  GESTION DE LA BASE DE DONNÉES:")
                print("  1. Mettre à jour le cycle en cours (rapide, recommandé)")
                print("  2. Réinitialiser la base de données (complet)")
                print("  3. Retour")
                
                sub_choice = input("\nVotre choix (1-3): ").strip()
                
                if sub_choice == "1":
                    print("\n🔄 Lancement de la mise à jour rapide...")
                    predictor.update_current_cycle()
                elif sub_choice == "2":
                    years = input("Nombre d'années de données à réinitialiser (défaut 3): ").strip() or "3"
                    years = int(years)
                    print(f"\n🗑️  Lancement de la réinitialisation complète avec {years} années...")
                    predictor.initialize_database(years)

            
            elif choice == "5":
                output_dir = input("Répertoire de sortie (défaut: .): ").strip() or "."
                print(f"\n📤 Export en cours vers {output_dir}...")
                visualizer.export_to_csv(output_dir)
            
            elif choice == "6":
                visualizer.display_statistics()
            
            elif choice == "7":
                print("\n🔬 Analyse du modèle (Importance des features)...")
                try:
                    features, target, feature_names = predictor.prepare_training_data()
                    _, X_test, _, y_test = train_test_split(
                        features, target, test_size=0.2, random_state=42, stratify=target
                    )
                    predictor.model = joblib.load('tempo_model.joblib')
                    predictor.display_feature_importance(X_train, X_test, y_test, feature_names, predictor.model.score(X_test, y_test))
                except FileNotFoundError:
                    print("❌ Le modèle 'tempo_model.joblib' n'a pas été trouvé. Veuillez d'abord l'entraîner (option 3).")
                except ValueError as e:
                    print(f"❌ Erreur lors de la préparation des données: {e}")
                except Exception as e:
                    print(f"❌ Une erreur est survenue: {e}")

            elif choice == "8":
                visualizer.view_performance_history()

            elif choice == "9":
                print("\n Au revoir!")
                break

            elif choice == "10":
                print("\n👋 Au revoir!")
                break

            elif choice == "0": # Hidden choice for old 9
                # Benchmark interactif
                print("\n🔬 BENCHMARK DES ALGORITHMES")
                print("Entrez les algorithmes à tester séparés par des espaces (ou laissez vide pour la liste par défaut)")
                print("Clés valides: mlp, random_forest, logistic, gb, svc")
                algos_input = input("Algorithmes [défaut: tous]: ").strip()
                if algos_input == "":
                    algos_list = None
                else:
                    algos_list = [a.strip() for a in algos_input.split() if a.strip()]

                cv_input = input("Nombre de folds CV (défaut 5): ").strip()
                try:
                    cv = int(cv_input) if cv_input else 5
                except ValueError:
                    print("Valeur invalide pour cv, utilisation de 5")
                    cv = 5

                n_jobs_input = input("Nombre de jobs pour le benchmark (défaut: -1 pour tous, 1 si debugger): ").strip()
                try:
                    n_jobs_val = int(n_jobs_input) if n_jobs_input != "" else None
                except ValueError:
                    print("Valeur invalide pour n_jobs, utilisation de None")
                    n_jobs_val = None

                print(f"Lancement du benchmark pour: {algos_list or 'tous'} avec cv={cv} et n_jobs={n_jobs_val}...")
                results = predictor.benchmark_algorithms(algorithms=algos_list, cv=cv, n_jobs=n_jobs_val)
                if results:
                    try:
                        import json
                        print(json.dumps(results, indent=2, ensure_ascii=False))
                    except Exception:
                        print(results)
            
            else:
                print("\n❌ Choix invalide. Veuillez choisir entre 1 et 10.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except ValueError as e:
            print(f"\n❌ Erreur de saisie: {e}")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
        
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()
