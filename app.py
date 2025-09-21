import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

app = Flask(__name__)

# -------------------------
# Load Real Estate Data
# -------------------------
df_real = None
for sep in [",", "\t", ";"]:
    temp = pd.read_csv("Real Estate.csv", sep=sep)
    if temp.shape[1] > 1:
        df_real = temp
        break

# Clean column names
df_real.columns = df_real.columns.str.strip().str.lower().str.replace(" ", "_")
df_real.rename(columns={
    "x1_transaction_date": "transaction_date",
    "x2_house_age": "house_age",
    "x3_distance_to_mrt_station": "mrt_distance",
    "x4_number_of_convenience_stores": "convenience_stores",
    "x5_latitude": "latitude",
    "x6_longitude": "longitude",
    "y_house_price_of_unit_area": "house_price"
}, inplace=True)

print("Renamed Columns in Real Estate CSV:")
print(df_real.columns.tolist())

X = df_real.drop("house_price", axis=1)
y = df_real["house_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

price_model = RandomForestRegressor(n_estimators=150, random_state=42)
price_model.fit(X_train, y_train)

y_pred = price_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------
# Load Flood dataset
# -------------------------
df_flood = pd.read_csv("flood.csv", sep="\t")
df_flood.columns = df_flood.columns.str.strip().str.lower().str.replace(" ", "_")

print("Renamed Columns in Flood CSV:")
print(df_flood.columns.tolist())

Xf = df_flood.drop("flood_risk", axis=1)
yf = df_flood["flood_risk"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)
flood_model = RandomForestClassifier(n_estimators=150, random_state=42)
flood_model.fit(Xf_train, yf_train)

yf_pred = flood_model.predict(Xf_test)
flood_acc = accuracy_score(yf_test, yf_pred)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_price():
    if request.method == "POST":
        # Convert date (YYYY-MM-DD) to numeric (e.g. year + month/100 for simplicity)
        transaction_date = request.form.get("transaction_date")
        year, month, _ = map(int, transaction_date.split("-"))
        transaction_date_num = year + month / 12.0

        # Collect other features
        house_age = float(request.form.get("house_age"))
        mrt_distance = float(request.form.get("mrt_distance"))
        convenience_stores = float(request.form.get("convenience_stores"))
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))

        # Arrange in correct order for model
        features = [
            transaction_date_num,
            house_age,
            mrt_distance,
            convenience_stores,
            latitude,
            longitude
        ]

        # Predict
        pred = price_model.predict([features])[0]

        return render_template(
            "predict.html",
            prediction=round(pred, 2),
            mse=mse,
            r2=r2
        )

    return render_template("predict.html")

@app.route("/flood", methods=["GET", "POST"])
def predict_flood():
    if request.method == "POST":
        features = [float(request.form.get(col)) for col in Xf.columns]
        pred = flood_model.predict([features])[0]
        risk = "High Flood Risk ⚠️" if pred == 1 else "Low Flood Risk ✅"
        return render_template("flood.html", prediction=risk, accuracy=flood_acc, Xf=Xf)
    return render_template("flood.html", Xf=Xf)

@app.route("/anomalies")
def detect_anomalies():
    preds = price_model.predict(X)
    residuals = y - preds
    df_real["predicted_price"] = preds
    df_real["error"] = residuals.abs()

    anomalies = df_real.sort_values("error", ascending=False).head(5)

    # Plot
    img = io.BytesIO()
    plt.figure(figsize=(6, 4))
    plt.scatter(y, preds, alpha=0.7, label="Data")
    plt.scatter(anomalies["house_price"], anomalies["predicted_price"], color="red", label="Anomalies")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.title("Predicted vs Actual with Anomalies Highlighted")
    plt.grid(True)
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("anomalies.html",
                           anomalies=anomalies[["house_age", "mrt_distance", "convenience_stores", "house_price", "predicted_price", "error"]].to_dict(orient="records"),
                           img_base64=plot_url)

@app.route("/analyze", methods=["POST"])
def analyze():
    issue = request.form.get("issue_description")
    root_cause_analysis = [
        "Why 1: Market demand fluctuations.",
        "Why 2: Location proximity to schools/transport.",
        "Why 3: Economic trends influencing real estate.",
        "Why 4: Data inconsistency in dataset.",
        "Why 5: Model limitations in capturing rare cases."
    ]
    return render_template("root_cause_analysis.html", issue_description=issue,
                           root_cause_analysis=root_cause_analysis, mse=mse, r2=r2)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
