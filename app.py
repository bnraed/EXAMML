from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import io
import base64
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'votre_clé_secrète'

# Identifiants simples
USERNAME = "admin"
PASSWORD = "admin"

# Chargement du modèle et des encodeurs
model = joblib.load('house_price_model.pkl')
le_location = joblib.load('label_encoder_location.pkl')
le_city = joblib.load('label_encoder_city.pkl')
le_governorate = joblib.load('label_encoder_governorate.pkl')

# Chargement du dataset
data = pd.read_csv('house_data_clean.csv')

def safe_transform(encoder, value):
    """
    Fonction de transformation sécurisée avec gestion d'erreur pour les encodeurs.
    """
    value_cleaned = value.strip().lower()
    encoder_classes_cleaned = [cls.strip().lower() for cls in encoder.classes_]

    if value_cleaned in encoder_classes_cleaned:
        index = encoder_classes_cleaned.index(value_cleaned)
        return encoder.transform([encoder.classes_[index]])[0]
    else:
        print(f"Valeur inconnue pour l'encodeur : '{value}', encodée par -1")
        return -1

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Route pour la page de connexion.
    """
    if 'user' in session:
        return redirect(url_for('index'))

    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['user'] = username
            flash('Connexion réussie!', 'success')
            return redirect(url_for('index'))
        else:
            error = "Identifiants incorrects"
            flash(error, 'danger')  # Affichage d'un message d'erreur

    return render_template('login.html', error=error)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """
    Route pour la déconnexion de l'utilisateur.
    """
    session.pop('user', None)
    flash('Déconnexion réussie!', 'success')  # Affichage d'un message de succès
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route principale de l'application, pour afficher le formulaire et les résultats.
    """
    if 'user' not in session:
        flash("Vous devez être connecté pour accéder à cette page.", 'warning')
        return redirect(url_for('login'))

    predicted_price = None
    img_data = {}

    # Valeurs par défaut pour les champs du formulaire
    location = city = governorate = area = pieces = room = bathroom = ''

    if request.method == 'POST':
        # Récupération des valeurs du formulaire
        location = request.form['location']
        city = request.form['city']
        governorate = request.form['governorate']
        area = request.form['area']
        pieces = request.form['pieces']
        room = request.form['room']
        bathroom = request.form['bathroom']

        # Encodage des données
        location_encoded = safe_transform(le_location, location)
        city_encoded = safe_transform(le_city, city)
        governorate_encoded = safe_transform(le_governorate, governorate)

        if location_encoded == -1 or city_encoded == -1 or governorate_encoded == -1:
            flash("L'une des valeurs saisies est inconnue. Veuillez vérifier vos entrées.", 'danger')
            return redirect(url_for('index'))

        # Préparation des données pour la prédiction
        new_data = pd.DataFrame([{
            'location_encoded': location_encoded,
            'city_encoded': city_encoded,
            'governorate_encoded': governorate_encoded,
            'area': float(area),
            'pieces': int(pieces),
            'room': int(room),
            'bathroom': int(bathroom)
        }])

        # Prédiction du prix
        predicted_price = model.predict(new_data)[0]

        # Création des graphiques
        fig1 = px.scatter(data, x="Area", y="price_tnd", color="governorate",
                          title="Relation entre Surface et Prix", labels={"Area": "Surface (m²)", "price_tnd": "Prix (TND)"})
        fig2 = px.histogram(data, x="price_tnd", nbins=30, title="Distribution des prix")
        fig3 = px.box(data, x="governorate", y="price_tnd", title="Distribution des prix par Gouvernorat")

        def fig_to_base64(fig):
            """
            Fonction pour convertir un graphique en image encodée en base64.
            """
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        # Encodage des graphiques
        img_data = {
            'scatter': fig_to_base64(fig1),
            'histogram': fig_to_base64(fig2),
            'boxplot': fig_to_base64(fig3)
        }

    return render_template('index.html',
                           predicted_price=predicted_price,
                           img_data=img_data,
                           location=location,
                           city=city,
                           governorate=governorate,
                           area=area,
                           pieces=pieces,
                           room=room,
                           bathroom=bathroom)

if __name__ == '__main__':
    app.run(debug=True)
