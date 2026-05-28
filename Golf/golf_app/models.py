from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))
    # This stores the benchmark the user is chasing
    target_handicap = db.Column(db.Integer, default=10)
    
    # relationship 'backref' allows us to call round.golfer to see who played it
    rounds = db.relationship('Round', backref='golfer', lazy=True, cascade="all, delete-orphan")

class Round(db.Model):
    __tablename__ = 'round'
    id = db.Column(db.Integer, primary_key=True)
    # Using utcnow ensures consistent timestamps regardless of server location
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    
    # Performance Metrics
    score = db.Column(db.Integer, nullable=False)
    gir = db.Column(db.Integer, default=0)            # Greens in Regulation (0-18)
    fir = db.Column(db.Integer, default=0)            # Fairways in Regulation (0-14)
    putts = db.Column(db.Integer, default=0)
    up_downs_att = db.Column(db.Integer, default=0)   # Scrambling Attempts
    up_downs_make = db.Column(db.Integer, default=0)  # Scrambling Success
    avg_drive_dist = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'<Round {self.score} by User {self.user_id}>'