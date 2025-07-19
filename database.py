import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

# Try to get database URL from Streamlit secrets if not in environment
if not DATABASE_URL:
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            DATABASE_URL = st.secrets.database.url
    except:
        pass

# Initialize database components only if URL is available
engine = None
SessionLocal = None
if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        print(f"Database engine creation failed: {e}")

Base = declarative_base()

class GameSession(Base):
    """Model for storing game sessions in the database."""
    __tablename__ = "game_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String(100), nullable=False)
    game_history = Column(Text, nullable=False)  # JSON string of game results
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_games = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class PredictionRecord(Base):
    """Model for storing prediction accuracy tracking."""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String(100), nullable=False)
    game_number = Column(Integer, nullable=False)  # Position in game history
    predicted_win_prob = Column(String(10), nullable=False)  # Stored as percentage string
    predicted_loss_prob = Column(String(10), nullable=False)
    predicted_draw_prob = Column(String(10), nullable=False)
    most_likely_outcome = Column(String(10), nullable=False)  # 'win', 'loss', or 'draw'
    actual_outcome = Column(Integer, nullable=False)  # 0=loss, 1=win, 2=draw
    was_correct = Column(Boolean, nullable=False)  # True if prediction matched actual
    prediction_confidence = Column(String(10), nullable=False)  # Max probability
    created_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    """Initialize the database and create tables."""
    if not DATABASE_URL or not engine:
        print("No database URL configured. App will run without database features.")
        return False
    
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")
        print("App will run without database features (no session persistence)")
        return False

def get_db_session():
    """Get a database session."""
    if not SessionLocal:
        return None
    
    try:
        db = SessionLocal()
        return db
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return None

def save_game_session(session_name: str, game_history: list):
    """Save a game session to the database."""
    db = get_db_session()
    if not db:
        return False
    
    try:
        # Calculate statistics
        wins = game_history.count(1)
        losses = game_history.count(0)
        draws = game_history.count(2)
        total_games = len(game_history)
        
        # Check if session already exists
        existing_session = db.query(GameSession).filter(
            GameSession.session_name == session_name,
            GameSession.is_active == True
        ).first()
        
        if existing_session:
            # Update existing session
            existing_session.game_history = json.dumps(game_history)
            existing_session.updated_at = datetime.utcnow()
            existing_session.total_games = total_games
            existing_session.wins = wins
            existing_session.losses = losses
            existing_session.draws = draws
        else:
            # Create new session
            new_session = GameSession(
                session_name=session_name,
                game_history=json.dumps(game_history),
                total_games=total_games,
                wins=wins,
                losses=losses,
                draws=draws
            )
            db.add(new_session)
        
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        st.error(f"Failed to save session: {str(e)}")
        return False
    finally:
        db.close()

def load_game_session(session_name: str):
    """Load a game session from the database."""
    db = get_db_session()
    if not db:
        return None
    
    try:
        session = db.query(GameSession).filter(
            GameSession.session_name == session_name,
            GameSession.is_active == True
        ).first()
        
        if session:
            game_history = json.loads(session.game_history)
            return {
                'game_history': game_history,
                'created_at': session.created_at,
                'updated_at': session.updated_at,
                'total_games': session.total_games,
                'wins': session.wins,
                'losses': session.losses,
                'draws': session.draws
            }
        return None
        
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return None
    finally:
        db.close()

def get_all_sessions():
    """Get all active game sessions."""
    db = get_db_session()
    if not db:
        return []
    
    try:
        sessions = db.query(GameSession).filter(
            GameSession.is_active == True
        ).order_by(GameSession.updated_at.desc()).all()
        
        return [{
            'name': session.session_name,
            'total_games': session.total_games,
            'wins': session.wins,
            'losses': session.losses,
            'draws': session.draws,
            'created_at': session.created_at,
            'updated_at': session.updated_at
        } for session in sessions]
        
    except Exception as e:
        st.error(f"Failed to load sessions: {str(e)}")
        return []
    finally:
        db.close()

def delete_game_session(session_name: str):
    """Delete a game session (soft delete by setting is_active to False)."""
    db = get_db_session()
    if not db:
        return False
    
    try:
        session = db.query(GameSession).filter(
            GameSession.session_name == session_name,
            GameSession.is_active == True
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
            return True
        return False
        
    except Exception as e:
        db.rollback()
        st.error(f"Failed to delete session: {str(e)}")
        return False
    finally:
        db.close()

def get_session_statistics():
    """Get overall statistics across all sessions."""
    db = get_db_session()
    if not db:
        return None
    
    try:
        sessions = db.query(GameSession).filter(GameSession.is_active == True).all()
        
        total_sessions = len(sessions)
        total_games = sum(session.total_games for session in sessions)
        total_wins = sum(session.wins for session in sessions)
        total_losses = sum(session.losses for session in sessions)
        total_draws = sum(session.draws for session in sessions)
        
        return {
            'total_sessions': total_sessions,
            'total_games': total_games,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_draws': total_draws,
            'overall_win_rate': total_wins / total_games if total_games > 0 else 0,
            'overall_loss_rate': total_losses / total_games if total_games > 0 else 0,
            'overall_draw_rate': total_draws / total_games if total_games > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Failed to get statistics: {str(e)}")
        return None
    finally:
        db.close()

def save_prediction_record(session_name: str, game_number: int, predictions: dict, actual_outcome: int):
    """Save a prediction record for accuracy tracking."""
    db = get_db_session()
    if not db:
        return False
    
    try:
        # Determine most likely outcome
        win_prob = predictions['win']['weighted_average']
        loss_prob = predictions['loss']['weighted_average']
        draw_prob = predictions['draw']['weighted_average']
        
        max_prob = max(win_prob, loss_prob, draw_prob)
        if max_prob == win_prob:
            most_likely = 'win'
            expected_value = 1
        elif max_prob == draw_prob:
            most_likely = 'draw'
            expected_value = 2
        else:
            most_likely = 'loss'
            expected_value = 0
        
        was_correct = (expected_value == actual_outcome)
        
        # Create prediction record
        prediction_record = PredictionRecord(
            session_name=session_name,
            game_number=game_number,
            predicted_win_prob=f"{win_prob:.1%}",
            predicted_loss_prob=f"{loss_prob:.1%}",
            predicted_draw_prob=f"{draw_prob:.1%}",
            most_likely_outcome=most_likely,
            actual_outcome=actual_outcome,
            was_correct=was_correct,
            prediction_confidence=f"{max_prob:.1%}"
        )
        
        db.add(prediction_record)
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        st.error(f"Failed to save prediction record: {str(e)}")
        return False
    finally:
        db.close()

def get_prediction_accuracy(session_name: str = None):
    """Get prediction accuracy statistics."""
    db = get_db_session()
    if not db:
        return None
    
    try:
        query = db.query(PredictionRecord)
        if session_name:
            query = query.filter(PredictionRecord.session_name == session_name)
        
        records = query.all()
        
        if not records:
            return None
        
        total_predictions = len(records)
        correct_predictions = sum(1 for record in records if record.was_correct)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Breakdown by outcome type
        win_predictions = [r for r in records if r.most_likely_outcome == 'win']
        loss_predictions = [r for r in records if r.most_likely_outcome == 'loss']
        draw_predictions = [r for r in records if r.most_likely_outcome == 'draw']
        
        win_accuracy = sum(1 for r in win_predictions if r.was_correct) / len(win_predictions) if win_predictions else 0
        loss_accuracy = sum(1 for r in loss_predictions if r.was_correct) / len(loss_predictions) if loss_predictions else 0
        draw_accuracy = sum(1 for r in draw_predictions if r.was_correct) / len(draw_predictions) if draw_predictions else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': accuracy,
            'win_predictions_count': len(win_predictions),
            'win_accuracy': win_accuracy,
            'loss_predictions_count': len(loss_predictions),
            'loss_accuracy': loss_accuracy,
            'draw_predictions_count': len(draw_predictions),
            'draw_accuracy': draw_accuracy,
            'recent_records': records[-10:]  # Last 10 predictions
        }
        
    except Exception as e:
        st.error(f"Failed to get prediction accuracy: {str(e)}")
        return None
    finally:
        db.close()

def get_all_prediction_accuracy():
    """Get overall prediction accuracy across all sessions."""
    db = get_db_session()
    if not db:
        return None
    
    try:
        records = db.query(PredictionRecord).all()
        
        if not records:
            return None
        
        total_predictions = len(records)
        correct_predictions = sum(1 for record in records if record.was_correct)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Group by session
        sessions = {}
        for record in records:
            if record.session_name not in sessions:
                sessions[record.session_name] = []
            sessions[record.session_name].append(record)
        
        session_accuracies = {}
        for session_name, session_records in sessions.items():
            session_correct = sum(1 for r in session_records if r.was_correct)
            session_total = len(session_records)
            session_accuracies[session_name] = session_correct / session_total if session_total > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': overall_accuracy,
            'session_count': len(sessions),
            'session_accuracies': session_accuracies,
            'best_session': max(session_accuracies.items(), key=lambda x: x[1]) if session_accuracies else None,
            'worst_session': min(session_accuracies.items(), key=lambda x: x[1]) if session_accuracies else None
        }
        
    except Exception as e:
        st.error(f"Failed to get overall prediction accuracy: {str(e)}")
        return None
    finally:
        db.close()