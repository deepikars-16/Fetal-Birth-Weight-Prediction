from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

basedir = os.path.abspath(os.path.dirname(__file__))

# ✅ Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for session management

# ✅ Initialize Database
db = SQLAlchemy()
db.init_app(app)

# ✅ Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ✅ Define Prediction Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_weight = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

# ✅ Create Database Tables
with app.app_context():
    db.create_all()

# ✅ Load ML Models and Preprocessing Tools
model = joblib.load("model/rf_model.pkl")
lr_model = joblib.load("model/lr_model.pkl")  # <-- Missing line added here
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")

# ✅ Registration Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
        else:
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")

# ✅ Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if not user:
            flash("Username not found. Please register.", "danger")
        elif not check_password_hash(user.password, password):
            flash("Invalid password. Try again!", "danger")
        else:
            session["user_id"] = user.id
            flash("Login successful!", "success")
            return redirect(url_for("predict"))

    return render_template("login.html")

# ✅ Logout Route
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))

# ✅ Prediction Route
@app.route("/", methods=["GET", "POST"])
def predict():
    rf_prediction = None
    lr_prediction = None

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            user_inputs = {
                "Age(years)": float(request.form["Age(years)"]),
                "Height(cm)": float(request.form["Height(cm)"]),
                "Parity": int(request.form["Parity"]),
                "ANC": int(request.form["ANC"]),
                "lwt(kg)": float(request.form["lwt(kg)"]),
                "FWt(kg)": float(request.form["FWt(kg)"]),
                "IBP_sys": int(request.form["IBP_sys"]),
                "IBP_dias": int(request.form["IBP_dias"]),
                "FBP_sys": int(request.form["FBP_sys"]),
                "FBP_dias": int(request.form["FBP_dias"]),
                "IHb(gm%)": float(request.form["IHb(gm%)"]),
                "FHb(gm%)": float(request.form["FHb(gm%)"]),
                "BS(RBS)": int(request.form["BS(RBS)"]),
                "LNH": int(request.form["LNH"]),
            }

            categorical_cols = ['Bgroup', 'Term/Preterm']
            for col in categorical_cols:
                value = request.form[col]
                for category in columns:
                    if category.startswith(col + "_"):
                        user_inputs[category] = 1 if category == f"{col}_{value}" else 0

            input_df = pd.DataFrame([user_inputs])
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0  

            input_df = input_df[columns]
            input_scaled = scaler.transform(input_df)

            # Predictions
            rf_prediction = round(model.predict(input_scaled)[0], 2)
            lr_prediction = round(lr_model.predict(input_scaled)[0], 2)

            # Store the prediction in the database
            new_prediction = Prediction(
                user_id=session["user_id"],
                predicted_weight=rf_prediction,
                date=datetime.strptime(request.form["date"], "%Y-%m-%d") if "date" in request.form else datetime.utcnow()
            )
            db.session.add(new_prediction)
            db.session.commit()

        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")

    return render_template("index.html", rf_prediction=rf_prediction, lr_prediction=lr_prediction)

# ✅ Graph Route
@app.route("/graph")
def show_graph():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_predictions = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.date).all()

    if not user_predictions:
        flash("No predictions available to display.", "warning")
        return redirect(url_for("predict"))

    dates = [p.date.strftime("%Y-%m-%d") for p in user_predictions]
    weights = [p.predicted_weight for p in user_predictions]

    return render_template("graph.html", dates=dates, weights=weights)

if __name__ == "__main__":
    app.run(debug=True)
