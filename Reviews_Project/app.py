from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = "secret123"

# ================= BASE DIRECTORY =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= DATABASE CONFIG =================
db_path = os.path.join(BASE_DIR, "users.db")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ================= MODELS =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    text = db.Column(db.String(1000))
    sentiment = db.Column(db.String(20))

# ================= LOAD CSV =================
csv_path = os.path.join(BASE_DIR, "product_reviews1.csv")

data = pd.read_csv(csv_path)

# clean dataset
data = data.dropna(subset=["CUSTOMER_REVIEW", "SENTIMENT"])

data["CUSTOMER_REVIEW"] = (
    data["CUSTOMER_REVIEW"]
    .astype(str)
    .str.lower()
)

data["SENTIMENT"] = data["SENTIMENT"].astype(int)

# ================= TRAIN MODEL =================
X = data["CUSTOMER_REVIEW"]
y = data["SENTIMENT"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=3000)

model.fit(X_vec, y)

# ================= SMART PREDICT =================
def smart_predict(text):

    text = text.lower().strip()

    # positive keywords
    positive_words = [
        "awesome", "excellent", "best", "amazing",
        "good", "great", "love", "perfect",
        "fantastic", "nice", "super", "smooth",
        "wonderful", "brilliant", "happy"
    ]

    # negative keywords
    negative_words = [
        "worst", "bad", "poor", "terrible",
        "waste", "useless", "slow", "lag",
        "horrible", "hate", "problem",
        "issue", "heating", "disappointed"
    ]

    # neutral keywords
    neutral_words = [
        "okay", "average", "fine",
        "decent", "not bad", "normal",
        "medium", "satisfactory"
    ]

    words = re.findall(r"\b\w+\b", text)

    # ================= RULE BASED =================

    # negative
    if any(word in text for word in negative_words):
        return "Negative 😞", 96

    # positive
    if any(word in text for word in positive_words):
        return "Positive 😊", 96

    # neutral
    if any(word in text for word in neutral_words):
        return "Neutral 😐", 90

    # ================= ML FALLBACK =================
    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    prob = model.predict_proba(vec).max()

    confidence = round(prob * 100, 2)

    # low confidence => neutral
    if confidence < 60:
        return "Neutral 😐", confidence

    if pred == 1:
        return "Positive 😊", confidence

    elif pred == -1:
        return "Negative 😞", confidence

    else:
        return "Neutral 😐", confidence

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def home():

    # login protection
    if "user" not in session:
        return redirect("/login")

    result = session.pop("result", None)
    confidence = session.pop("confidence", 0)
    top_words = session.pop("top_words", [])

    # ================= REVIEW ANALYSIS =================
    if request.method == "POST":

        review = request.form["review"].strip()

        if review:

            # prevent duplicate on reload
            if session.get("last_review") == review:
                return redirect(url_for("home"))

            # predict sentiment
            result, confidence = smart_predict(review)

            # ================= KEYWORDS =================
            vec = vectorizer.transform([review])

            feature_names = vectorizer.get_feature_names_out()

            scores = vec.toarray()[0]

            top_idx = np.argsort(scores)[-5:]

            top_words = [
                feature_names[i]
                for i in top_idx
                if scores[i] > 0
            ]

            # ================= SAVE REVIEW =================
            new_review = Review(
                username=session["user"],
                text=review,
                sentiment=result
            )

            db.session.add(new_review)
            db.session.commit()

            # save session data
            session["last_review"] = review
            session["result"] = result
            session["confidence"] = confidence
            session["top_words"] = top_words

            return redirect(url_for("home"))

    # ================= REVIEW HISTORY =================
    history = (
        Review.query
        .filter_by(username=session["user"])
        .order_by(Review.id.desc())
        .all()
    )

    # ================= CHART COUNTS =================
    pos_count = sum(
        1 for h in history
        if "Positive" in h.sentiment
    )

    neg_count = sum(
        1 for h in history
        if "Negative" in h.sentiment
    )

    neu_count = sum(
        1 for h in history
        if "Neutral" in h.sentiment
    )

    # ================= PRODUCT STATS =================
    product_stats = []

    for product in data["PRODUCT_NAME"].dropna().unique():

        subset = data[data["PRODUCT_NAME"] == product]

        total = len(subset)

        pos = len(subset[subset["SENTIMENT"] == 1])

        neg = len(subset[subset["SENTIMENT"] == -1])

        neu = len(subset[subset["SENTIMENT"] == 0])

        rating = round(3 + ((pos - neg) / total) * 2, 1)

        product_stats.append({
            "name": product,
            "positive": round((pos / total) * 100, 1),
            "negative": round((neg / total) * 100, 1),
            "neutral": round((neu / total) * 100, 1),
            "rating": rating
        })

    # ================= BEST PRODUCT =================
    best_product = max(
        product_stats,
        key=lambda x: x["positive"]
    )["name"]

    # ================= PRODUCT LINKS =================
    product_links = {
        "IPHONE 11": "https://www.apple.com",
        "ALEXA": "https://www.amazon.in",
        "LENOVO LEGION 5": "https://www.lenovo.com",
        "ONEPLUS NORD": "https://www.oneplus.in",
        "REALME 11": "https://www.realme.com",
        "BOAT HEADPHONES": "https://www.boat-lifestyle.com",
        "SAMSUNG TV": "https://www.samsung.com"
    }

    # ================= RENDER =================
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        top_words=top_words,
        history=history[:5],
        products=product_stats,
        best_product=best_product,
        product_links=product_links,
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count
    )

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"].strip()

        password = generate_password_hash(
            request.form["password"]
        )

        # check existing user
        existing_user = User.query.filter_by(
            username=username
        ).first()

        if existing_user:
            return "User already exists"

        # create user
        new_user = User(
            username=username,
            password=password
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")

# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user"] = user.username

            return redirect("/")

        return "Invalid Credentials"

    return render_template("login.html")

# ================= LOGOUT =================
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")

# ================= CREATE DATABASE =================
with app.app_context():
    db.create_all()

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
