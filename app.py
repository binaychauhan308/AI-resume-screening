import os
import re
import sqlite3
from functools import wraps

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
SIMPLE_AUTH_USERNAME = os.environ.get("APP_USERNAME", "admin")
SIMPLE_AUTH_PASSWORD = os.environ.get("APP_PASSWORD", "admin123")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Predefined Engineering Jobs ---
PREDEFINED_JOBS = {
    "software_engineer": "Software Engineer role requiring strong programming in Python, JavaScript, Java, or C++. Must be skilled in Agile workflows. Experience with databases and version control (Git) is required.",
    "data_scientist": "Data Scientist position requiring expertise in Python, R, and SQL. Must have deep knowledge of Machine Learning, NLP, TensorFlow, and PyTorch. Experience handling large datasets and scikit-learn.",
    "ai_engineer": "AI/ML Engineer requiring advanced Python and C++ skills. Experience in Deep Learning, building predictive models, and deploying AI solutions using Docker and AWS.",
    "devops_engineer": "DevOps Engineer focused on infrastructure and automation. Requires strong skills in Bash, Python, or Go. Must have deep expertise in CI/CD, Jenkins, Docker, Kubernetes, Linux, and cloud platforms like AWS or Azure.",
    "cloud_engineer": "Cloud Engineer responsible for designing and maintaining cloud infrastructure. Requires strong knowledge of AWS, GCP, or Azure. Familiarity with infrastructure as code, Kubernetes, and Docker.",
    "frontend_engineer": "Frontend Web Developer requiring expert-level HTML, CSS, JavaScript, and TypeScript. Must be highly skilled in modern frameworks like React. Passionate about UI/UX and responsive design.",
    "backend_engineer": "Backend Developer focused on server-side logic and APIs. Requires expertise in Python (Django/Flask), Java (Spring Boot), Node.js, or Go. Strong knowledge of SQL databases and server architecture.",
    "fullstack_engineer": "Full Stack Engineer capable of handling both frontend and backend. Requires JavaScript/TypeScript, React, Node.js, and Python. Experience with SQL databases, Git, and deploying web applications.",
    "data_engineer": "Data Engineer responsible for building data pipelines. Requires advanced SQL and Python skills. Experience with big data tools, ETL processes, and cloud storage (AWS, GCP).",
    "qa_engineer": "QA Automation Engineer focused on software testing. Requires programming skills in Python or Java. Must have experience with automated testing frameworks like Selenium or Cypress, and CI/CD pipelines.",
}

PREDEFINED_ROLE_SKILLS = {
    "software_engineer": ["Python", "JavaScript", "Java", "C++", "Agile", "Databases", "Git"],
    "data_scientist": ["Python", "R", "SQL", "Machine Learning", "NLP", "TensorFlow", "PyTorch", "scikit-learn"],
    "ai_engineer": ["Python", "C++", "Deep Learning", "Docker", "AWS"],
    "devops_engineer": ["Bash", "Python", "Go", "CI/CD", "Jenkins", "Docker", "Kubernetes", "Linux", "AWS", "Azure"],
    "cloud_engineer": ["AWS", "GCP", "Azure", "Infrastructure as Code", "Kubernetes", "Docker"],
    "frontend_engineer": ["HTML", "CSS", "JavaScript", "TypeScript", "React", "UI/UX", "Responsive Design"],
    "backend_engineer": ["Python", "Django", "Flask", "Java", "Spring Boot", "Node.js", "Go", "SQL", "APIs"],
    "fullstack_engineer": ["JavaScript", "TypeScript", "React", "Node.js", "Python", "SQL", "Git"],
    "data_engineer": ["SQL", "Python", "ETL", "Data Pipelines", "AWS", "GCP"],
    "qa_engineer": ["Python", "Java", "Selenium", "Cypress", "Automated Testing", "CI/CD"],
}


def get_db_connection():
    conn = sqlite3.connect("resumes.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS candidates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  candidate_name TEXT, filename TEXT, resume_text TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS auth_users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    except Exception as error:
        print(f"Error: {error}")
    return text


def clean_text(text):
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", text).lower().split())


def rank_resumes(job_description, resumes_list):
    if not resumes_list:
        return []
    cleaned_jd = clean_text(job_description)
    cleaned_resumes = [clean_text(resume) for resume in resumes_list]
    documents = [cleaned_jd] + cleaned_resumes
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()


def normalize_skill(skill):
    return " ".join(skill.lower().split())


def build_required_skills(selected_role, required_skills_input):
    if required_skills_input.strip():
        raw_skills = [skill.strip() for skill in required_skills_input.split(",") if skill.strip()]
    else:
        raw_skills = PREDEFINED_ROLE_SKILLS.get(selected_role, [])

    skills = []
    seen = set()
    for skill in raw_skills:
        normalized = normalize_skill(skill)
        if normalized not in seen:
            seen.add(normalized)
            skills.append({"label": skill, "normalized": normalized})
    return skills


def skill_in_text(resume_text, normalized_skill):
    escaped_skill = re.escape(normalized_skill).replace(r"\ ", r"\s+")
    pattern = rf"(?<![a-z0-9]){escaped_skill}(?![a-z0-9])"
    return re.search(pattern, resume_text.lower()) is not None


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def get_safe_next_path(next_path):
    if next_path and next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return None


def get_auth_user(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM auth_users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


@app.context_processor
def inject_auth_state():
    return {
        "is_authenticated": bool(session.get("user_id")),
        "current_user_name": session.get("user_name"),
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    next_path = get_safe_next_path(request.args.get("next"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        next_path = get_safe_next_path(request.form.get("next"))

        if not username or not password:
            flash("Please fill in all fields.")
        elif username == SIMPLE_AUTH_USERNAME.lower():
            flash("That username is reserved. Please choose another one.")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long.")
        elif password != confirm_password:
            flash("Passwords do not match.")
        elif get_auth_user(username):
            flash("That username already exists.")
        else:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
            conn.commit()
            conn.close()

            session["user_id"] = f"user:{username}"
            session["user_name"] = username
            flash("Account created successfully.")
            return redirect(next_path or url_for("dashboard"))

    return render_template("register.html", next_path=next_path or "")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    next_path = get_safe_next_path(request.args.get("next"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        next_path = get_safe_next_path(request.form.get("next"))

        if username == SIMPLE_AUTH_USERNAME.lower() and password == SIMPLE_AUTH_PASSWORD:
            session["user_id"] = SIMPLE_AUTH_USERNAME
            session["user_name"] = SIMPLE_AUTH_USERNAME
            flash(f"Welcome, {SIMPLE_AUTH_USERNAME}!")
            return redirect(next_path or url_for("dashboard"))

        user = get_auth_user(username)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.")
        else:
            session["user_id"] = f"user:{user['id']}"
            session["user_name"] = user["username"]
            flash(f"Welcome, {user['username']}!")
            return redirect(next_path or url_for("dashboard"))

    return render_template("login.html", next_path=next_path or "")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("home"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    ranked_candidates = []
    job_description = ""
    selected_role = "custom"
    required_skills_input = ""
    required_skill_labels = []

    if request.method == "POST":
        selected_role = request.form.get("job_role", "custom")
        required_skills_input = request.form.get("required_skills", "")

        if selected_role in PREDEFINED_JOBS:
            job_description = PREDEFINED_JOBS[selected_role]
        else:
            job_description = request.form.get("job_description", "")

        required_skills = build_required_skills(selected_role, required_skills_input)
        required_skill_labels = [skill["label"] for skill in required_skills]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates")
        db_candidates = cursor.fetchall()
        conn.close()

        if db_candidates:
            resumes_text = [row["resume_text"] for row in db_candidates]
            scores = rank_resumes(job_description, resumes_text)

            for index, row in enumerate(db_candidates):
                candidate_data = dict(row)
                score = round(scores[index] * 100, 2)
                candidate_data["match_score"] = score

                if required_skills:
                    resume_text = candidate_data["resume_text"] or ""
                    matched = [
                        skill["label"]
                        for skill in required_skills
                        if skill_in_text(resume_text, skill["normalized"])
                    ]
                    missing = [
                        skill["label"]
                        for skill in required_skills
                        if skill["label"] not in matched
                    ]

                    candidate_data["matched_skills"] = ", ".join(matched) if matched else "None"
                    candidate_data["missing_skills"] = ", ".join(missing) if missing else "None"
                    candidate_data["skill_score"] = round((len(matched) / len(required_skills)) * 100)
                    candidate_data["job_match_label"] = (
                        f"{candidate_data['skill_score']}% ({len(matched)}/{len(required_skills)} skills)"
                    )
                    candidate_data["status"] = "Shortlisted" if candidate_data["skill_score"] >= 50 else "Rejected"
                else:
                    candidate_data["matched_skills"] = "Not provided"
                    candidate_data["missing_skills"] = "Not provided"
                    candidate_data["skill_score"] = -1
                    candidate_data["job_match_label"] = "Not available"
                    candidate_data["status"] = "Shortlisted" if score >= 20.0 else "Rejected"

                ranked_candidates.append(candidate_data)

            if required_skills:
                ranked_candidates = sorted(
                    ranked_candidates,
                    key=lambda candidate: (candidate["skill_score"], candidate["match_score"]),
                    reverse=True,
                )
            else:
                ranked_candidates = sorted(
                    ranked_candidates,
                    key=lambda candidate: candidate["match_score"],
                    reverse=True,
                )

    return render_template(
        "index.html",
        candidates=ranked_candidates,
        jd=job_description,
        role=selected_role,
        req_skills=required_skills_input,
        required_skill_labels=required_skill_labels,
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload_resumes():
    files = request.files.getlist("resume_files")
    if not files or files[0].filename == "":
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()
    success_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            resume_text = extract_text_from_pdf(filepath)
            candidate_name = filename.rsplit(".", 1)[0].replace("_", " ")

            cursor.execute(
                "INSERT INTO candidates (candidate_name, filename, resume_text) VALUES (?, ?, ?)",
                (candidate_name, filename, resume_text),
            )
            success_count += 1

    conn.commit()
    conn.close()
    flash(f"Successfully processed {success_count} resumes.")
    return redirect(url_for("dashboard"))


@app.route("/candidate/<int:candidate_id>")
@login_required
def view_candidate(candidate_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    candidate = cursor.fetchone()
    conn.close()

    if not candidate:
        flash("Candidate not found.")
        return redirect(url_for("dashboard"))

    filename = candidate["filename"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        flash("Resume PDF file was not found.")
        return redirect(url_for("dashboard"))

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        mimetype="application/pdf",
        as_attachment=False,
    )


init_db()


if __name__ == "__main__":
    app.run(debug=True)
