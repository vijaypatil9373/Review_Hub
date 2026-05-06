from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = "secret123"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# ================= MODELS =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    text = db.Column(db.String(500))
    sentiment = db.Column(db.String(20))

# ================= LOAD DATA =================
data = pd.read_csv("product_reviews1.csv")

data = data.dropna(subset=["CUSTOMER_REVIEW", "SENTIMENT"])
data["CUSTOMER_REVIEW"] = data["CUSTOMER_REVIEW"].str.lower()
data["SENTIMENT"] = data["SENTIMENT"].astype(int)

X = data["CUSTOMER_REVIEW"]
y = data["SENTIMENT"]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_vec, y)

# ================= SMART PREDICT =================
def smart_predict(text):
    text = text.lower()

    positive_words = ["awesome","excellent","best","amazing","good","great","love","perfect"]
    negative_words = ["worst","bad","poor","terrible","waste","useless","slow","lag"]
    neutral_words  = ["okay","average","fine","decent","not bad"]

    words = re.findall(r'\b\w+\b', text)

    if any(w in words for w in negative_words):
        return "Negative 😞", 95

    if any(w in words for w in positive_words):
        return "Positive 😊", 95

    if any(w in text for w in neutral_words):
        return "Neutral 😐", 90

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    confidence = round(prob * 100, 2)

    if confidence < 60:
        return "Neutral 😐", confidence

    if pred == 1:
        return "Positive 😊", confidence
    elif pred == -1:
        return "Negative 😞", confidence
    else:
        return "Neutral 😐", confidence

# ================= HOME =================
@app.route("/", methods=["GET","POST"])
def home():
    if "user" not in session:
        return redirect("/login")

    result = session.pop("result", None)
    confidence = session.pop("confidence", 0)
    top_words = session.pop("top_words", [])

    if request.method == "POST":
        review = request.form["review"].strip()

        if review:

            # 🚫 Prevent reload duplicate (same last review)
            if session.get("last_review") == review:
                return redirect(url_for("home"))

            result, confidence = smart_predict(review)

            # keywords
            vec = vectorizer.transform([review])
            feature_names = vectorizer.get_feature_names_out()
            scores = vec.toarray()[0]
            top_idx = np.argsort(scores)[-5:]
            top_words = [feature_names[i] for i in top_idx if scores[i] > 0]

            # ✅ ALWAYS SAVE (duplicates allowed)
            db.session.add(Review(
                username=session["user"],
                text=review,
                sentiment=result
            ))
            db.session.commit()

            # ✅ store last review to prevent reload duplicate
            session["last_review"] = review

            session["result"] = result
            session["confidence"] = confidence
            session["top_words"] = top_words

            return redirect(url_for("home"))

    # ================= HISTORY =================
    history = Review.query.filter_by(username=session["user"])\
                          .order_by(Review.id.desc()).all()

    pos_count = sum(1 for h in history if "Positive" in h.sentiment)
    neg_count = sum(1 for h in history if "Negative" in h.sentiment)
    neu_count = sum(1 for h in history if "Neutral" in h.sentiment)

    # ================= PRODUCT STATS =================
    product_stats = []
    for product in data["PRODUCT_NAME"].unique():
        subset = data[data["PRODUCT_NAME"] == product]
        total = len(subset)

        pos = len(subset[subset["SENTIMENT"] == 1])
        neg = len(subset[subset["SENTIMENT"] == -1])
        neu = len(subset[subset["SENTIMENT"] == 0])

        product_stats.append({
            "name": product,
            "positive": round(pos/total*100,1),
            "negative": round(neg/total*100,1),
            "neutral": round(neu/total*100,1),
            "rating": round(3 + (pos-neg)/total*2,1)
        })

    best_product = max(product_stats, key=lambda x: x["positive"])["name"]

    product_links = {
        "IPHONE 11": "https://www.apple.com",
        "ALEXA": "https://www.amazon.in",
        "LENOVO LEGION 5": "https://www.lenovo.com",
        "ONEPLUS NORD": "https://www.oneplus.in",
        "REALME 11": "https://www.realme.com",
        "BOAT HEADPHONES": "https://www.boat-lifestyle.com",
        "SAMSUNG TV": "https://www.samsung.com"
    }

    return render_template("index.html",
        result=result,
        confidence=confidence,
        top_words=top_words,
        history=history[:5],   # latest 5 only
        products=product_stats,
        best_product=best_product,
        product_links=product_links,
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count
    )

# ================= AUTH =================
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            return "User already exists"

        db.session.add(User(username=username, password=password))
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if user and check_password_hash(user.password, request.form["password"]):
            session["user"] = user.username
            return redirect("/")
        return "Invalid Credentials"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)