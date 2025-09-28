from flask import Flask, render_template, request
import sqlite3, pickle, numpy as np
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image

DB_PATH = "face_auth.db"
TABLE = "users"
MODEL_NAME = "Facenet"
THRESHOLD = 0.5

app = Flask(__name__)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (username TEXT PRIMARY KEY, embedding BLOB)")
    conn.commit()
    conn.close()

def save_embedding(username, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"INSERT OR REPLACE INTO {TABLE} VALUES (?,?)", (username, pickle.dumps(embedding)))
    conn.commit()
    conn.close()

def load_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT username, embedding FROM {TABLE}")
    data = [(u, pickle.loads(e)) for u, e in c.fetchall()]
    conn.close()
    return data

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def decode_base64_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    message = ""
    if request.method == "POST":
        username = request.form.get("username").strip()
        img_data = request.form.get("img_data")
        if img_data:
            frame = decode_base64_image(img_data)
            try:
                rep = DeepFace.represent(frame, model_name=MODEL_NAME, enforce_detection=True)
                encoding = rep[0]["embedding"]
                save_embedding(username, encoding)
                message = f"User '{username}' registered successfully!"
            except Exception as e:
                message = f"Failed to detect face: {e}"
        else:
            message = "No image received."
    return render_template("register.html", message=message)

@app.route("/login", methods=["GET","POST"])
def login():
    message = ""
    if request.method == "POST":
        img_data = request.form.get("img_data")
        if img_data:
            frame = decode_base64_image(img_data)
            try:
                query_enc = DeepFace.represent(frame, model_name=MODEL_NAME, enforce_detection=True)[0]["embedding"]
                entries = load_embeddings()
                best_user, best_dist = None, 1e9
                for u, emb in entries:
                    dist = cosine_distance(query_enc, emb)
                    if dist < best_dist:
                        best_dist, best_user = dist, u
                if best_dist < THRESHOLD:
                    message = f"Logged in as {best_user} ✅"
                else:
                    message = "Face not recognized ❌"
            except Exception as e:
                message = f"Failed to detect face: {e}"
        else:
            message = "No image received."
    return render_template("login.html", message=message)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
