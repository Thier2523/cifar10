# ================= IMPORTS =================
from io import BytesIO
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import base64

# ================= CONFIG =================
#os.chdir(r"C:\Users\tsow2\OneDrive\Bureau\cours2iemeannee\Projets\Projet personnel\Projet2_cnn")

app = Flask(__name__)
model = load_model("model_cnn.keras")

# Classes CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ================= HOME =================
@app.route("/")
def Home():

    # Message d’accueil précisant les classes reconnues
    welcome_message = (
        f"Bienvenue dans le prédicteur d'images conçu par Thierno Djiby Sow ! "
        f"Les images analysées doivent appartenir aux classes suivantes : "
        f"{', '.join(classes)}."
    )

    return render_template("index.html", welcome_message=welcome_message)

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    files = request.files.getlist("file")
    if not files or files[0].filename == '':
        return render_template("index.html",
                               prediction_text="Aucun fichier sélectionné")

    results = []

    for file in files:

        file_bytes = file.read()

        # Prétraitement
        img = image.load_img(BytesIO(file_bytes), target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction
        preds = model.predict(img_array)[0]
        predicted_class = classes[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Probabilités
        probabilities = {classes[i]: float(preds[i]) for i in range(len(classes))}

        message = f"Prédiction pour cette image : {predicted_class} ({confidence:.1f}%)"

        # Encode image pour HTML
        img_base64 = base64.b64encode(file_bytes).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_base64}"

        results.append({
            "message": message,
            "img_url": img_url,
            "probabilities": probabilities
        })

    group_message = f"Analyse d’un groupe de {len(results)} image(s) :"

    return render_template(
        "index.html",
        results=results,
        group_message=group_message
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
