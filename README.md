Face Recognition Authentication App
===================================

A lightweight web application to register and log in users using facial recognition. 
Built with Flask and DeepFace, it stores user embeddings in a SQLite database.

Features
--------
- Register users with a face image.
- Log in users via face recognition.
- Uses DeepFace (Facenet) for facial embeddings.
- Simple HTML frontend.
- SQLite database for storing embeddings.

Requirements
------------
- Python 3.10+
- Flask
- DeepFace
- Pillow
- NumPy

Install dependencies:
---------------------
pip install flask deepface pillow numpy

Usage
-----
1. Clone the repository:
   git clone <repo_url>
   cd <repo_folder>

2. Run the app:
   python app.py

3. Open in browser:
   http://127.0.0.1:5000/

Database
--------
- `face_auth.db` (auto-created)
- Stores users in table `users` with columns:
  - username (TEXT PRIMARY KEY)
  - embedding (BLOB)

How It Works
------------
1. Registration:
   - User provides username and face image.
   - Face embedding is computed and saved in DB.

2. Login:
   - User provides face image.
   - Embedding is compared with DB entries using cosine distance.
   - If distance < 0.5, user is authenticated.

Notes
-----
- Use clear, frontal face images for best accuracy.
- Threshold for recognition can be adjusted in app.py.

License
-------
MIT License
