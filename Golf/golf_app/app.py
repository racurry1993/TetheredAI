from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Round
import os
import urllib.parse
from datetime import timedelta, datetime, time
import sqlalchemy

app = Flask(__name__)

# --- 1. CONFIGURATION ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'golf-dev-secret-123')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

PROJECT_ID = "amateurgolfer" 
DATASET_ID = "performance_data"
KEY_FILE = 'amateurgolfer-4cb540c6b99b.json'
KEY_PATH = os.path.join(os.path.dirname(__file__), KEY_FILE)

db_user = "postgres" 
db_pass = urllib.parse.quote_plus('Z%<"0"#5v=KxDFEe')
db_name = "performance_data"
instance_connection_name = "amateurgolfer:us-central1:golferhandicap"
public_ip = "104.154.56.28"

if os.environ.get('GAE_ENV') == 'standard' or os.environ.get('K_SERVICE'):
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"postgresql+pg8000://{db_user}:{db_pass}@/{db_name}"
        f"?unix_sock=/cloudsql/{instance_connection_name}/.s.PGSQL.5432"
    )
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql+psycopg2://{db_user}:{db_pass}@{public_ip}/{db_name}"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"DATABASE ERROR: {e}")

# --- 2. HELPERS ---

def get_engine():
    try:
        from engine import GolfEngine
        csv_path = os.path.join(os.path.dirname(__file__), 'Handicap Stats.csv')
        return GolfEngine(csv_path) if os.path.exists(csv_path) else None
    except:
        return None

def get_performance_trends(rounds, target_hcp):
    if not rounds:
        return {'dates': [], 'scores': [], 'gir': [], 'putts': [], 'fir': [], 'drive_dist': [], 'benchmarks': {}}
    
    recent = rounds[:10][::-1]
    engine = get_engine()
    raw_targets = engine.get_benchmark_stats(target_hcp) if engine else {}
    
    # Normalizing benchmarks to ensure they match user percentage scales (0-100)
    benchmarks = {
        'scores': target_hcp + 72,
        'gir': raw_targets.get('GIR', 0) * 100 if raw_targets.get('GIR', 0) < 1 else raw_targets.get('GIR', 0),
        'fir': raw_targets.get('FIR', 0) * 100 if raw_targets.get('FIR', 0) < 1 else raw_targets.get('FIR', 0),
        'putts': raw_targets.get('Putts per Rd', 0),
        'dist': raw_targets.get('Avg Drive Dist', 0)
    }

    return {
        'dates': [r.date.strftime('%m/%d') for r in recent],
        'scores': [r.score for r in recent],
        'gir': [(r.gir/18)*100 for r in recent],
        'putts': [r.putts for r in recent],
        'fir': [(r.fir/14)*100 for r in recent],
        'dist': [r.avg_drive_dist for r in recent],
        'benchmarks': benchmarks
    }

# --- 3. ROUTES ---

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            hashed_pw = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
            new_user = User(
                email=request.form['email'], 
                name=request.form['name'], 
                password=hashed_pw,
                target_handicap=int(request.form.get('target_hcp', 10))
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except:
            db.session.rollback()
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    rounds = Round.query.filter_by(user_id=current_user.id).order_by(Round.date.desc()).all()
    
    user_avgs = {'GIR':0, 'FIR':0, 'Putts per Rd':0, 'Up & Down':0, 'Avg Drive Dist':0}
    priorities, target_stats = [], {}
    
    if rounds:
        count = len(rounds)
        user_avgs = {
            'GIR': (sum(r.gir for r in rounds) / (count * 18)) * 100,
            'FIR': (sum(r.fir for r in rounds) / (count * 14)) * 100,
            'Putts per Rd': sum(r.putts for r in rounds) / count,
            'Up & Down': (sum(r.up_downs_make for r in rounds) / max(1, sum(r.up_downs_att for r in rounds))) * 100,
            'Avg Drive Dist': sum(r.avg_drive_dist for r in rounds) / count
        }
        
        engine = get_engine()
        if engine:
            # Map categories to search terms for the JavaScript popup
            drill_keywords = {
                'GIR': 'iron play accuracy',
                'FIR': 'driver accuracy fairway',
                'Up & Down': 'chipping and pitching',
                'Putts per Rd': 'putting distance control',
                'Avg Drive Dist': 'increase driving distance'
            }

            priorities = engine.get_priorities(user_avgs, current_user.target_handicap)
            
            for p in priorities:
                # Normalization Fix: If engine target is 0.4, convert to 40.0 
                # to ensure (User 2.3 < Target 40.0) triggers correctly
                if p['category'] in ['GIR', 'FIR', 'Up & Down'] and p['target'] < 1:
                    p['target'] = p['target'] * 100
                
                # Recalculate the gap based on normalized values
                if p['category'] == 'Putts per Rd':
                    p['gap'] = p['user'] - p['target']
                else:
                    p['gap'] = p['target'] - p['user']

                # Attach search query for the JS modal
                p['search_query'] = drill_keywords.get(p['category'], 'golf swing')

            raw_targets = engine.get_benchmark_stats(current_user.target_handicap)
            target_stats = {
                'GIR': raw_targets.get('GIR', 0) * 100 if raw_targets.get('GIR', 0) < 1 else raw_targets.get('GIR', 0),
                'FIR': raw_targets.get('FIR', 0) * 100 if raw_targets.get('FIR', 0) < 1 else raw_targets.get('FIR', 0),
                'Up & Down': raw_targets.get('Up & Down', 0) * 100 if raw_targets.get('Up & Down', 0) < 1 else raw_targets.get('Up & Down', 0),
                'Putts per Rd': raw_targets.get('Putts per Rd', 0),
                'Avg Drive Dist': raw_targets.get('Avg Drive Dist', 0)
            }

    return render_template('dashboard.html', 
                           priorities=priorities, 
                           user_avgs=user_avgs, 
                           target_stats=target_stats,
                           rounds=rounds,
                           trends=get_performance_trends(rounds, current_user.target_handicap),
                           correlations={
                               'score_gir': [{'x': (r.gir/18)*100, 'y': r.score} for r in rounds], 
                               'score_putts': [{'x': r.putts, 'y': r.score} for r in rounds],
                               'score_fir': [{'x': (r.fir/14)*100, 'y': r.score} for r in rounds]
                           } if rounds else {},
                           today=datetime.utcnow().strftime('%Y-%m-%d'))

@app.route('/add_round_batch', methods=['POST'])
@login_required
def add_round_batch():
    dates = request.form.getlist('date[]')
    scores = request.form.getlist('score[]')
    girs = request.form.getlist('gir[]')
    firs = request.form.getlist('fir[]')
    putts = request.form.getlist('putts[]')
    ud_atts = request.form.getlist('ud_att[]')
    ud_makes = request.form.getlist('ud_make[]')
    dists = request.form.getlist('dist[]')

    for i in range(len(scores)):
        if not scores[i]: continue
        new_round = Round(
            user_id=current_user.id,
            date=datetime.strptime(dates[i], '%Y-%m-%d'),
            score=int(scores[i]),
            gir=int(girs[i] or 0),
            fir=int(firs[i] or 0),
            putts=int(putts[i] or 0),
            up_downs_att=int(ud_atts[i] or 0),
            up_downs_make=int(ud_makes[i] or 0),
            avg_drive_dist=float(dists[i] or 0)
        )
        db.session.add(new_round)
    db.session.commit()
    flash("Rounds analyzed and added!")
    return redirect(url_for('dashboard'))

@app.route('/update_target', methods=['POST'])
@login_required
def update_target():
    current_user.target_handicap = int(request.form.get('target_hcp'))
    db.session.commit()
    flash("Target updated!")
    return redirect(url_for('dashboard'))

@app.route('/delete_round/<int:id>')
@login_required
def delete_round(id):
    round_to_delete = Round.query.get(id)
    if round_to_delete.user_id == current_user.id:
        db.session.delete(round_to_delete)
        db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/edit_round/<int:id>', methods=['POST'])
@login_required
def edit_round(id):
    r = Round.query.get(id)
    if r.user_id == current_user.id:
        r.date = datetime.strptime(request.form.get('date'), '%Y-%m-%d')
        r.score = int(request.form.get('score'))
        r.gir = int(request.form.get('gir'))
        r.fir = int(request.form.get('fir'))
        r.putts = int(request.form.get('putts'))
        r.up_downs_att = int(request.form.get('ud_att'))
        r.up_downs_make = int(request.form.get('ud_make'))
        r.avg_drive_dist = float(request.form.get('dist'))
        db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)