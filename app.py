from flask import Flask ,render_template,request
import joblib
import pandas as pd

app= Flask(__name__)

model = joblib.load("model/fraud_detection_model.pkl")

# Load Label Encoder
encoder = joblib.load("model/label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")    

@app.route("/predict",methods=["POST"])
def predict():
    step = int(request.form["step"])

    transaction_type = request.form["type"]

    amount = float(request.form["amount"])

    oldbalanceOrg = float(request.form["oldbalanceOrg"])

    newbalanceOrig = float(request.form["newbalanceOrig"])

    oldbalanceDest = float(request.form["oldbalanceDest"])

    newbalanceDest = float(request.form["newbalanceDest"])
    
    # Encode Transaction Type
    transaction_type = encoder.transform([transaction_type])[0]

    # Create Input Data
    input_data = pd.DataFrame([{
    "step": step,
    "type": transaction_type,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest
}])
    # Predict
    prediction = model.predict(input_data)[0]

    # Prediction Confidence
    probability = model.predict_proba(input_data)[0]
    print("Prediction:", prediction)
    print("Probability:", probability)
    # Display Result and Confidence
    if prediction == 1:
        result = "⚠️ Fraudulent Transaction Detected"
        confidence = round(probability[1] * 100, 2)
        card_class = "danger"

    else:
        result = "✅ Legitimate Transaction"
        confidence = round(probability[0] * 100, 2)
        card_class = "success"

    return render_template(
    "index.html",
    prediction=result,
    confidence=confidence,
    card_class=card_class
)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5200 , debug=True)
    