from database.schema import SessionLocal, UserProgress, PracticeSession, Mudra
from datetime import datetime

class ProgressTracker:
    def __init__(self, user_id=1):
        self.user_id = user_id
        self.session_data = {
            "attempts": {},
            "start_time": datetime.utcnow()
        }

    def record_attempt(self, predicted_mudra_name, confidence, is_correct):
        """
        Record a single frame's attempt during a live session.
        Since video runs at 30fps, we only call this if we have a stable prediction for > 1 second.
        """
        if predicted_mudra_name not in self.session_data["attempts"]:
            self.session_data["attempts"][predicted_mudra_name] = {
                "total": 0,
                "correct": 0,
                "confidence_sum": 0.0
            }
            
        data = self.session_data["attempts"][predicted_mudra_name]
        data["total"] += 1
        if is_correct:
            data["correct"] += 1
        data["confidence_sum"] += confidence

    def end_session(self):
        """
        Saves session data to SQLite and updates aggregate UserProgress.
        """
        if not self.session_data["attempts"]:
            return

        db = SessionLocal()
        try:
            # 1. Update UserProgress (aggregate statistics)
            total_session_correct = 0
            total_session_attempts = 0
            
            for mudra_name, stats in self.session_data["attempts"].items():
                mudra = db.query(Mudra).filter(Mudra.name == mudra_name).first()
                if not mudra:
                    continue
                    
                total_session_correct += stats["correct"]
                total_session_attempts += stats["total"]
                
                # Fetch or create UserProgress
                up = db.query(UserProgress).filter(
                    UserProgress.user_id == self.user_id,
                    UserProgress.mudra_id == mudra.id
                ).first()
                
                avg_confidence = stats["confidence_sum"] / max(1, stats["total"])
                
                if not up:
                    up = UserProgress(
                        user_id=self.user_id,
                        mudra_id=mudra.id,
                        total_attempts=stats["total"],
                        correct_attempts=stats["correct"],
                        avg_confidence=avg_confidence
                    )
                    db.add(up)
                else:
                    new_total = up.total_attempts + stats["total"]
                    new_correct = up.correct_attempts + stats["correct"]
                    
                    # Moving average for confidence
                    old_sum = up.avg_confidence * up.total_attempts
                    new_sum = old_sum + stats["confidence_sum"]
                    up.avg_confidence = new_sum / new_total
                    
                    up.total_attempts = new_total
                    up.correct_attempts = new_correct
                    
            # 2. Record PracticeSession
            end_time = datetime.utcnow()
            duration_minutes = (end_time - self.session_data["start_time"]).total_seconds() / 60.0
            overall_accuracy = (total_session_correct / max(1, total_session_attempts)) * 100
            
            new_session = PracticeSession(
                user_id=self.user_id,
                start_time=self.session_data["start_time"],
                duration_minutes=duration_minutes,
                total_mudras_practiced=len(self.session_data["attempts"]),
                overall_accuracy=overall_accuracy
            )
            db.add(new_session)
            db.commit()
            
        except Exception as e:
            db.rollback()
            print(f"Error saving session: {e}")
        finally:
            db.close()
            
        # Reset session data
        self.session_data = {"attempts": {}, "start_time": datetime.utcnow()}
