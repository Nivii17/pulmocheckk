from flask import Flask, request
import joblib, pandas as pd, sqlite3
app = Flask(__name__)
model = joblib.load("model.pkl")
con = sqlite3.connect("pulmo.db", check_same_thread=False)

HISTORY = ["Smoking Status", "Alcohol Use", "Family History of Lung Disease", "Past Lung Condition",
           "Air Pollution Exposure", "Occupational Exposure", "Allergies"]
SYMPTOMS = ["Cough Days", "Wheezing", "Shortness of Breath", "Chest Tightness", "Fatigue"]
TESTS = ["Oxygen Saturation", "Lung Function FEV1 %", "BMI"]

def flags(p):
    f = [("Current smoker", p["Smoking Status"] == "Current"), ("Family history of lung disease", p["Family History of Lung Disease"] == "Yes"),
         (f"Past lung condition ({p['Past Lung Condition']})", p["Past Lung Condition"] != "No history"),
         ("High air pollution exposure", p["Air Pollution Exposure"] == "High"), ("Occupational exposure", p["Occupational Exposure"] == "Yes"),
         (f"Long cough ({p['Cough Days']} days)", p["Cough Days"] > 14), ("Wheezing", p["Wheezing"] == "Yes"),
         ("Shortness of breath", p["Shortness of Breath"] == "Yes"), (f"Low oxygen ({p['Oxygen Saturation']}%)", p["Oxygen Saturation"] < 94),
         (f"Low lung function (FEV1 {p['Lung Function FEV1 %']}%)", p["Lung Function FEV1 %"] < 70)]
    return [name for name, hit in f if hit]

def show(p, cols):
    return "<ul>" + "".join(f"<li>{c}: {p[c]}</li>" for c in cols) + "</ul>"

@app.route("/")
def home():
    pid = request.args.get("id", "").strip().upper()
    page = '<h2>PulmoCheck</h2><form>Patient ID (P1 - P1000): <input name="id"><button>Predict</button></form>'
    if pid:
        row = pd.read_sql('SELECT * FROM patients WHERE "Patient Id" = ?', con, params=(pid,))
        if row.empty:
            return page + "<p>Patient not found</p>"
        p = row.iloc[0]
        X = row[list(model.feature_names_in_)]
        risk, conf = model.predict(X)[0], model.predict_proba(X).max()
        page += f"<p>Patient {pid} | Age: {p['Age']} | Gender: {p['Gender']}</p>"
        page += f"<h3>{risk} risk ({conf:.0%} confidence)</h3><p><b>Main risk factors:</b> {', '.join(flags(p)) or 'None'}</p>"
        page += f"<h4>Patient history</h4>{show(p, HISTORY)}<h4>Current symptoms</h4>{show(p, SYMPTOMS)}<h4>Test readings</h4>{show(p, TESTS)}"
    return page

if __name__ == "__main__":
    app.run()
