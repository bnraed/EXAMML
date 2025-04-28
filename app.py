from flask import Flask, render_template, request
import pandas as pd
import joblib
import io
import base64
import plotly.express as px

app = Flask(__name__)

# Chargement du modèle et des encodeurs
model = joblib.load('house_price_model.pkl')
le_location = joblib.load('label_encoder_location.pkl')
le_city = joblib.load('label_encoder_city.pkl')
le_governorate = joblib.load('label_encoder_governorate.pkl')

# Chargement du dataset pour les stats
data = pd.read_csv('house_data_clean.csv')

# Fonction de transformation sécurisée
def safe_transform(encoder, value):
    value_cleaned = value.strip().lower()
    encoder_classes_cleaned = [cls.strip().lower() for cls in encoder.classes_]

    if value_cleaned in encoder_classes_cleaned:
        index = encoder_classes_cleaned.index(value_cleaned)
        return encoder.transform([encoder.classes_[index]])[0]
    else:
        print(f"Valeur inconnue pour l'encodeur : '{value}', encodée par -1")
        return -1

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    img_data = None

    # Initialiser les champs vides pour le premier chargement
    location = ''
    city = ''
    governorate = ''
    area = ''
    pieces = ''
    room = ''
    bathroom = ''

    if request.method == 'POST':
        location = request.form['location']
        city = request.form['city']
        governorate = request.form['governorate']
        area = request.form['area']
        pieces = request.form['pieces']
        room = request.form['room']
        bathroom = request.form['bathroom']

        # Encodage
        location_encoded = safe_transform(le_location, location)
        city_encoded = safe_transform(le_city, city)
        governorate_encoded = safe_transform(le_governorate, governorate)

        new_data = pd.DataFrame([{
            'location_encoded': location_encoded,
            'city_encoded': city_encoded,
            'governorate_encoded': governorate_encoded,
            'area': float(area),
            'pieces': int(pieces),
            'room': int(room),
            'bathroom': int(bathroom)
        }])

        predicted_price = model.predict(new_data)[0]

        # Générer plusieurs graphiques
        fig1 = px.scatter(data, x="Area", y="price_tnd", color="governorate",
                          title="Relation entre Surface et Prix", labels={"Area": "Surface (m²)", "price_tnd": "Prix (TND)"})
        fig2 = px.histogram(data, x="price_tnd", nbins=30, title="Distribution des prix")
        fig3 = px.box(data, x="governorate", y="price_tnd", title="Distribution des prix par Gouvernorat")

        img1 = io.BytesIO()
        fig1.write_image(img1, format='png')
        img1.seek(0)
        img_data1 = base64.b64encode(img1.read()).decode('utf-8')

        img2 = io.BytesIO()
        fig2.write_image(img2, format='png')
        img2.seek(0)
        img_data2 = base64.b64encode(img2.read()).decode('utf-8')

        img3 = io.BytesIO()
        fig3.write_image(img3, format='png')
        img3.seek(0)
        img_data3 = base64.b64encode(img3.read()).decode('utf-8')

        img_data = {
            'scatter': img_data1,
            'histogram': img_data2,
            'boxplot': img_data3
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
