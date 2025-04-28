import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger les données
data = pd.read_csv('house_data.csv')

# Standardiser les noms de colonnes (en minuscules)
data.columns = data.columns.str.strip().str.lower()

# Vérifie les colonnes pour t'assurer qu'on est bon
print("Colonnes après normalisation :", data.columns)

# Liste des colonnes nécessaires
required_cols = ['location', 'city', 'governorate', 'area', 'pieces', 'room', 'bathroom', 'price_tnd']

# Supprimer les lignes qui ont des valeurs manquantes dans ces colonnes
data = data.dropna(subset=required_cols)

# Préparation des features
X = data[['location', 'city', 'governorate', 'area', 'pieces', 'room', 'bathroom']]
y = data['price_tnd']

# Encodage des variables catégorielles
le_location = LabelEncoder()
le_city = LabelEncoder()
le_governorate = LabelEncoder()

X['location_encoded'] = le_location.fit_transform(X['location'])
X['city_encoded'] = le_city.fit_transform(X['city'])
X['governorate_encoded'] = le_governorate.fit_transform(X['governorate'])

# On garde uniquement les colonnes numériques pour l'entraînement
X = X[['location_encoded', 'city_encoded', 'governorate_encoded', 'area', 'pieces', 'room', 'bathroom']]

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle et des encodeurs
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(le_location, 'label_encoder_location.pkl')
joblib.dump(le_city, 'label_encoder_city.pkl')
joblib.dump(le_governorate, 'label_encoder_governorate.pkl')

print("✅ Modèle et encodeurs entraînés et sauvegardés avec succès.")
print(data.head())
