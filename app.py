import os
import sqlite3
from flask import (
    Flask, jsonify, request, render_template,
    redirect, url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------- Certificates (existing services.py) ----------------
from services import (
    expiring_within,
    total_active,
    predicted_failures_count,
    predict_failure,
    certs_import_from_csv,
    certs_list,
    certs_add,
    certs_update,
    certs_delete,
)
from fitness_certificates_ai import upsert_from_csv as certs_upsert

# ---------------- Job Cards (DB; ML optional) ----------------
from jobcard_status_solution import (
    upsert_from_csv as jc_upsert,
    list_jobcards,
    add_jobcard,
    update_jobcard,
    delete_jobcard,
    summary_for_train,
)

# ML is optional — don’t crash if not present
try:
    from jobcard_status_solution import (
        train_duration_model, risk_for_open_jobs, predict_close_days
    )
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    def train_duration_model(*a, **k): return {"trained": 0, "error": "ML not available"}
    def risk_for_open_jobs(limit=500): return {"rows": [], "count": 0, "note": "ML not available"}
    def predict_close_days(x): return {"error": "ML not available"}

# ---------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("APP_SECRET", "dev-secret-change-me")

# ---------------------------------------------------------------------
# Users DB (login/signup)
# ---------------------------------------------------------------------
USER_DB = os.path.join("data", "users.db")
os.makedirs("data", exist_ok=True)

def init_user_db():
    """Create users table and add columns if they’re missing."""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Add optional columns if not there yet
    c.execute("PRAGMA table_info(users)")
    cols = {row[1] for row in c.fetchall()}
    to_add = []
    if "first_name" not in cols: to_add.append(("first_name", "TEXT"))
    if "last_name"  not in cols: to_add.append(("last_name", "TEXT"))
    if "email"      not in cols: to_add.append(("email", "TEXT UNIQUE"))
    for name, typ in to_add:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {name} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()

init_user_db()

def _read_any(keys):
    """Read keys from JSON or form; trim strings."""
    out = {}
    if request.is_json:
        src = request.get_json(silent=True) or {}
        for k in keys:
            v = src.get(k)
            out[k] = v.strip() if isinstance(v, str) else v
    else:
        for k in keys:
            v = request.form.get(k, "")
            out[k] = v.strip() if isinstance(v, str) else v
    return out

def _alias(*names, default=""):
    """Return first non-empty alias value from JSON or form."""
    if request.is_json:
        src = request.get_json(silent=True) or {}
        for n in names:
            v = src.get(n)
            if isinstance(v, str): v = v.strip()
            if v: return v
    else:
        for n in names:
            v = request.form.get(n, "")
            if isinstance(v, str): v = v.strip()
            if v: return v
    return default

# ---------------------------------------------------------------------
# Auth routes (match your login.html tabs)
# ---------------------------------------------------------------------
@app.route("/login.html", methods=["GET", "POST"])
@app.route("/login", methods=["POST"])
def login_page():
    if request.method == "POST":
        data = _read_any(["username", "password"])
        username, password = data.get("username", ""), data.get("password", "")
        if not username or not password:
            msg = "Username and password are required"
            if request.is_json: return jsonify({"error": msg}), 400
            flash(msg, "danger"); return render_template("login.html")

        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[1], password):
            session["user_id"] = row[0]
            session["username"] = username
            if request.is_json: return jsonify({"ok": True})
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            if request.is_json: return jsonify({"error": "Invalid username or password"}), 401
            flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/signup.html", methods=["GET", "POST"])
@app.route("/signup", methods=["POST"])
def signup_page():
    if request.method == "POST":
        # Accept common field names from your UI
        first_name = _alias("first_name", "firstname", "fname")
        last_name  = _alias("last_name", "lastname", "lname")
        email      = _alias("email", "mail")
        username   = _alias("username", "user", "uid")
        password   = _alias("password", "pass", "pwd")
        confirm    = _alias("confirm_password", "confirmPassword", "password2", "cpassword", "confirm")

        if not username or not email or not password:
            msg = "Username, email and password are required"
            if request.is_json: return jsonify({"error": msg}), 400
            flash(msg, "danger"); return render_template("login.html")

        if password != confirm:
            msg = "Passwords do not match"
            if request.is_json: return jsonify({"error": msg}), 400
            flash(msg, "danger"); return render_template("login.html")

        try:
            hashed = generate_password_hash(password)
            conn = sqlite3.connect(USER_DB)
            c = conn.cursor()
            c.execute("""
                INSERT INTO users (username, password, first_name, last_name, email)
                VALUES (?, ?, ?, ?, ?)
            """, (username, hashed, first_name, last_name, email))
            conn.commit()
            conn.close()
            if request.is_json: return jsonify({"ok": True})
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login_page"))
        except sqlite3.IntegrityError:
            msg = "Username or email already exists"
            if request.is_json: return jsonify({"error": msg}), 409
            flash(msg, "danger")
        except Exception as e:
            msg = f"Server error: {e}"
            if request.is_json: return jsonify({"error": msg}), 500
            flash("Server error. Try again later.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login_page"))

# ---------------------------------------------------------------------
# Seed business DBs on startup (idempotent)
# ---------------------------------------------------------------------
try:
    certs_upsert("data/fitness_certificates_history.csv")
except Exception as e:
    app.logger.info(f"Certificates seed skipped: {e}")
try:
    jc_upsert("data/jobcards.csv")
except Exception as e:
    app.logger.info(f"Jobcards seed skipped: {e}")

# ---------------------------------------------------------------------
# Page routes (unchanged)
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fitnesscertificate.html")
def fitness_page():
    return render_template("fitnesscertificate.html")

@app.route("/aboutus.html")
def about_page():
    return render_template("aboutus.html")

@app.route("/certs.html")
def certs_admin_page():
    return render_template("certs.html")

@app.route("/jobcards.html")
def jobcards_admin_page():
    return render_template("jobcards.html")

@app.route("/jobcard.html")
def jobcard_dashboard():
    return render_template("jobcard.html")

@app.route("/brandingpriorities.html")
def branding_priorities_page():
    return render_template("brandingpriorities.html")

@app.route("/stablinggeometry.html")
def stabling_geometry_page():
    return render_template("stablinggeometry.html")

# ---------------------------------------------------------------------
# Certificates APIs (unchanged)
# ---------------------------------------------------------------------
@app.route("/api/expiring")
def api_expiring():
    days = request.args.get("days", default=7, type=int)
    return jsonify(expiring_within(days))

@app.route("/api/total")
def api_total():
    return jsonify(total_active())

@app.route("/api/predicted_failures")
def api_predicted_failures():
    return jsonify(predicted_failures_count())

@app.route("/api/predict_failure")
def api_predict_failure():
    return jsonify(predict_failure())

@app.route("/api/certs/import", methods=["POST"])
def api_certs_import():
    res = certs_import_from_csv()
    return (jsonify(res), 200) if "error" not in res else (jsonify(res), 400)

@app.route("/api/certs", methods=["GET"])
def api_certs_list():
    limit = request.args.get("limit", default=200, type=int)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(certs_list(limit, offset))

@app.route("/api/certs", methods=["POST"])
def api_certs_add():
    payload = request.get_json(silent=True) or {}
    return jsonify(certs_add(payload))

@app.route("/api/certs/<int:cert_id>", methods=["PATCH"])
def api_certs_update(cert_id):
    patch = request.get_json(silent=True) or {}
    return jsonify(certs_update(cert_id, patch))

@app.route("/api/certs/<int:cert_id>", methods=["DELETE"])
def api_certs_delete(cert_id):
    return jsonify(certs_delete(cert_id))

# ---------------------------------------------------------------------
# Job Cards APIs (unchanged)
# ---------------------------------------------------------------------
@app.route("/api/jobcards/import", methods=["POST"])
def api_jobcards_import():
    return jsonify(jc_upsert("data/jobcards.csv"))

@app.route("/api/jobcards", methods=["GET"])
def api_jobcards_list():
    limit = request.args.get("limit", default=1000, type=int)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(list_jobcards(limit, offset))

@app.route("/api/jobcards", methods=["POST"])
def api_jobcards_add():
    payload = request.get_json(silent=True) or {}
    return jsonify(add_jobcard(payload))

@app.route("/api/jobcards/<int:jc_id>", methods=["PATCH"])
def api_jobcards_update(jc_id):
    patch = request.get_json(silent=True) or {}
    return jsonify(update_jobcard(jc_id, patch))

@app.route("/api/jobcards/<int:jc_id>", methods=["DELETE"])
def api_jobcards_delete(jc_id):
    return jsonify(delete_jobcard(jc_id))

@app.route("/api/jobstatus")
def api_jobstatus():
    train_id = request.args.get("train_id", type=int)
    return jsonify(summary_for_train(train_id))

# ---------------------------------------------------------------------
# Job Cards ML APIs (unchanged; resilient)
# ---------------------------------------------------------------------
@app.route("/api/jobcards/ml/train", methods=["POST"])
def api_jobcards_ml_train():
    use_csv = bool(request.args.get("use_csv", "0") == "1")
    res = train_duration_model(use_csv=use_csv)
    return (jsonify(res), 501) if "error" in res and "not available" in res["error"] else jsonify(res)

@app.route("/api/jobcards/ml/risk")
def api_jobcards_ml_risk():
    limit = request.args.get("limit", default=500, type=int)
    res = risk_for_open_jobs(limit)
    return (jsonify(res), 501) if "note" in res and "not available" in res["note"] else jsonify(res)

@app.route("/api/jobcards/<int:jc_id>/predict")
def api_jobcard_predict(jc_id):
    res = predict_close_days(jc_id)
    return (jsonify(res), 501) if "error" in res and "not available" in res["error"] else jsonify(res)

# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting Flask on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
