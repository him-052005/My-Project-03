from __future__ import annotations
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models import init_db, SessionLocal, User, Student, Attendance, ensure_day_rows
from face_utils import capture_samples, train_or_update_model, recognize_once, FACES_DIR, DATA_DIR

APP_TZ = ZoneInfo("Asia/Kolkata")
BASE_BACKEND = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(DATA_DIR, 'exports')
os.makedirs(EXPORTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_BACKEND, 'templates'), static_folder=os.path.join(BASE_BACKEND, 'static'))
app.config['SECRET_KEY'] = 'super-secret-change-me'

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class AuthUser(UserMixin):
    def __init__(self, db_user: User):
        self.id = str(db_user.id)
        self.username = db_user.username

@login_manager.user_loader
def load_user(user_id):
    with SessionLocal() as s:
        u = s.get(User, int(user_id))
        if not u:
            return None
        return AuthUser(u)

# ---------- One-time DB bootstrap (admin/admin123) ----------
with SessionLocal() as s:
    init_db()
    admin = s.query(User).filter_by(username='admin').one_or_none()
    if not admin:
        s.add(User(username='admin', password_hash=generate_password_hash('admin123')))
        s.commit()

# ================= ROUTES =================
@app.route('/')
@login_required
def dashboard():
    today = datetime.now(APP_TZ).date()
    with SessionLocal() as s:
        ensure_day_rows(s, today)
        total = s.query(Student).count()
        present = s.query(Attendance).filter_by(day=today, status='Present').count()
    return render_template('dashboard.html', total=total, present=present, day=today.strftime('%Y-%m-%d'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        with SessionLocal() as s:
            u = s.query(User).filter_by(username=username).one_or_none()
            if u and check_password_hash(u.password_hash, password):
                login_user(AuthUser(u))
                return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------- Students ----------
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        roll_no = request.form['roll_no'].strip()
        name = request.form['name'].strip()
        email = request.form.get('email', '').strip()
        klass = request.form.get('klass', '').strip()
        section = request.form.get('section', '').strip()
        with SessionLocal() as s:
            exists = s.query(Student).filter_by(roll_no=roll_no).one_or_none()
            if exists:
                flash('Roll No already exists', 'warning')
                return redirect(url_for('register'))
            st = Student(roll_no=roll_no, name=name, email=email, klass=klass, section=section)
            s.add(st)
            s.commit()
            # capture samples
            preview = capture_samples(st.id, samples=20)
            if preview:
                st.image_path = preview
                s.commit()
            trained = train_or_update_model()
            flash('Student registered and model updated' if trained else 'Student registered (insufficient data to train yet)', 'success')
            return redirect(url_for('students'))
    return render_template('register.html')

@app.route('/students')
@login_required
def students():
    with SessionLocal() as s:
        rows = s.query(Student).order_by(Student.roll_no).all()
    return render_template('students.html', rows=rows, faces_dir=os.path.relpath(FACES_DIR, BASE_BACKEND))

@app.route('/student/<int:sid>/edit', methods=['GET', 'POST'])
@login_required
def student_edit(sid):
    with SessionLocal() as s:
        st = s.get(Student, sid)
        if not st:
            flash('Student not found', 'danger')
            return redirect(url_for('students'))
        if request.method == 'POST':
            st.roll_no = request.form['roll_no'].strip()
            st.name = request.form['name'].strip()
            st.email = request.form.get('email', '').strip()
            st.klass = request.form.get('klass', '').strip()
            st.section = request.form.get('section', '').strip()
            if request.form.get('recapture') == 'on':
                preview = capture_samples(st.id, samples=15)
                if preview:
                    st.image_path = preview
            s.commit()
            train_or_update_model()
            flash('Student updated', 'success')
            return redirect(url_for('students'))
    return render_template('student_edit.html', st=st)

@app.route('/download_students')
@login_required
def download_students():
    with SessionLocal() as s:
        rows = s.query(Student).order_by(Student.roll_no).all()
    df = pd.DataFrame([
        {
            'Roll No': r.roll_no,
            'Name': r.name,
            'Email': r.email,
            'Class': r.klass,
            'Section': r.section,
        } for r in rows
    ])
    out_path = os.path.join(EXPORTS_DIR, 'students.xlsx')
    df.to_excel(out_path, index=False)
    return send_file(out_path, as_attachment=True)

# ---------- Attendance ----------
@app.route('/attendance')
@login_required
def attendance_page():
    today = datetime.now(APP_TZ).date()
    with SessionLocal() as s:
        ensure_day_rows(s, today)
        rows = (
            s.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .filter(Attendance.day == today)
            .order_by(Student.roll_no)
            .all()
        )
    table = [
        {
            'roll_no': stu.roll_no,
            'name': stu.name,
            'status': att.status,
            'timestamp': att.timestamp.strftime('%H:%M:%S') if att.timestamp else ''
        } for att, stu in rows
    ]
    return render_template('attendance.html', today=today.strftime('%Y-%m-%d'), table=table)

@app.route('/api/mark_attendance', methods=['POST'])
@login_required
def api_mark_attendance():
    sid = recognize_once(threshold=70.0)
    if sid is None:
        return jsonify({'ok': False, 'msg': 'No confident face match'}), 400
    today = datetime.now(APP_TZ).date()
    now = datetime.now(APP_TZ)
    with SessionLocal() as s:
        att = s.query(Attendance).filter_by(student_id=sid, day=today).one_or_none()
        if not att:
            att = Attendance(student_id=sid, day=today, status='Absent')
            s.add(att); s.commit()
        att.status = 'Present'
        att.timestamp = now
        s.commit()
        stu = s.get(Student, sid)
        return jsonify({'ok': True, 'roll_no': stu.roll_no, 'name': stu.name, 'time': now.strftime('%H:%M:%S')})

@app.route('/api/attendance_today')
@login_required
def api_attendance_today():
    today = datetime.now(APP_TZ).date()
    with SessionLocal() as s:
        rows = (
            s.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .filter(Attendance.day == today)
            .order_by(Student.roll_no)
            .all()
        )
    data = [
        {
            'Roll No': stu.roll_no,
            'Name': stu.name,
            'Status': att.status,
            'Timestamp': att.timestamp.strftime('%H:%M:%S') if att.timestamp else ''
        } for att, stu in rows
    ]
    present = sum(1 for r in data if r['Status'] == 'Present')
    return jsonify({'total': len(data), 'present': present, 'absent': len(data)-present, 'rows': data})

@app.route('/download_attendance')
@login_required
def download_attendance_today():
    d = datetime.now(APP_TZ).date()
    return _download_attendance_for_date(d)

@app.route('/download_attendance_by_date')
@login_required
def download_attendance_by_date():
    dstr = request.args.get('date')
    d = datetime.strptime(dstr, '%Y-%m-%d').date()
    return _download_attendance_for_date(d)

@app.route('/download_attendance_pdf')
@login_required
def download_attendance_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    dstr = request.args.get('date')
    d = datetime.now(APP_TZ).date() if not dstr else datetime.strptime(dstr, '%Y-%m-%d').date()
    with SessionLocal() as s:
        rows = (
            s.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .filter(Attendance.day == d)
            .order_by(Student.roll_no)
            .all()
        )
    pdf_path = os.path.join(EXPORTS_DIR, f"attendance_{d}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, f"Attendance – {d}")
    y = height-90
    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Roll No")
    c.drawString(150, y, "Name")
    c.drawString(350, y, "Status")
    c.drawString(430, y, "Time")
    y -= 20
    for att, stu in rows:
        if y < 60:
            c.showPage(); y = height-60
        c.drawString(50, y, stu.roll_no or '')
        c.drawString(150, y, stu.name or '')
        c.drawString(350, y, att.status)
        c.drawString(430, y, att.timestamp.strftime('%H:%M:%S') if att.timestamp else '')
        y -= 18
    c.showPage(); c.save()
    return send_file(pdf_path, as_attachment=True)

# helper

def _download_attendance_for_date(d: date):
    with SessionLocal() as s:
        rows = (
            s.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .filter(Attendance.day == d)
            .order_by(Student.roll_no)
            .all()
        )
    df = pd.DataFrame([
        {
            'Roll No': stu.roll_no,
            'Name': stu.name,
            'Status': att.status,
            'Timestamp': att.timestamp.strftime('%H:%M:%S') if att.timestamp else ''
        } for att, stu in rows
    ])
    out_path = os.path.join(EXPORTS_DIR, f'attendance_{d}.xlsx')
    df.to_excel(out_path, index=False)
    return send_file(out_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)