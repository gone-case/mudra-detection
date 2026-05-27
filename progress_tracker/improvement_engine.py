from database.schema import SessionLocal, UserProgress, Mudra
from sqlalchemy import desc

class ImprovementEngine:
    def __init__(self, user_id=1):
        self.user_id = user_id

    def get_weak_mudras(self, limit=3):
        """
        Identify mudras where the user struggles (accuracy < 70% and at least 5 attempts)
        or mudras with lowest accuracy.
        """
        db = SessionLocal()
        try:
            progress_records = db.query(UserProgress).filter(
                UserProgress.user_id == self.user_id,
                UserProgress.total_attempts > 0
            ).all()
            
            # Calculate accuracy for each
            mudra_stats = []
            for record in progress_records:
                accuracy = (record.correct_attempts / record.total_attempts) * 100
                mudra = db.query(Mudra).filter(Mudra.id == record.mudra_id).first()
                if mudra:
                    mudra_stats.append({
                        "mudra_name": mudra.name,
                        "accuracy": accuracy,
                        "total_attempts": record.total_attempts,
                        "common_mistakes": mudra.common_mistakes,
                        "improvement_tips": mudra.improvement_tips
                    })
                    
            # Sort by lowest accuracy first
            mudra_stats.sort(key=lambda x: x["accuracy"])
            
            # Return top N weak mudras
            # Filtering out those with very few attempts so we don't punish 0/1 false starts too early,
            # but if all are few attempts, we just return the lowest.
            weak_mudras = mudra_stats[:limit]
            return weak_mudras
        finally:
            db.close()

    def generate_recommendations(self):
        """
        Generate actionable suggestions string array.
        """
        weak_mudras = self.get_weak_mudras(limit=3)
        suggestions = []
        
        if not weak_mudras:
            return ["Great job! You don't have any weak mudras yet. Keep practicing new ones!"]
            
        for stat in weak_mudras:
            if stat["accuracy"] < 50:
                suggestions.append(f"🔴 You seem to be struggling with **{stat['mudra_name']}** ({stat['accuracy']:.1f}% accuracy). Tip: {stat['improvement_tips']}")
            elif stat["accuracy"] < 80:
                suggestions.append(f"🟠 You can improve your **{stat['mudra_name']}** ({stat['accuracy']:.1f}% accuracy). Remember: {stat['common_mistakes']}")
            else:
                suggestions.append(f"🟢 **{stat['mudra_name']}** is solid ({stat['accuracy']:.1f}%), but you can still polish it! Tip: {stat['improvement_tips']}")
                
        return suggestions
