import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AlertService:
    def __init__(self):
        self.alert_history = []
        self.alert_cooldown = 30  # seconds
        
    def check_alerts(self, objects, situation_analysis):
        """Check for alert conditions"""
        alerts = []
        
        # Check for unusual people count
        people_count = situation_analysis['people_count']
        if people_count > 10:
            alerts.append(self._create_alert(
                "CROWD_DETECTED",
                f"Unusually large crowd detected: {people_count} people",
                "medium"
            ))
        
        # Check for specific objects of concern
        concerning_objects = ['weapon', 'knife', 'gun', 'fire']
        for obj in objects:
            if any(concerning in obj['class'] for concerning in concerning_objects):
                alerts.append(self._create_alert(
                    "CONCERNING_OBJECT",
                    f"Concerning object detected: {obj['class']}",
                    "high"
                ))
        
        # Check for unusual activity patterns
        if situation_analysis['activity_level'] == 'high' and people_count == 0:
            alerts.append(self._create_alert(
                "UNUSUAL_ACTIVITY",
                "High activity detected with no people present",
                "medium"
            ))
        
        # Check for abandoned objects
        if self._check_abandoned_objects(objects):
            alerts.append(self._create_alert(
                "ABANDONED_OBJECT",
                "Possible abandoned object detected",
                "low"
            ))
        
        # Filter alerts by cooldown
        filtered_alerts = self._filter_alerts_by_cooldown(alerts)
        
        return filtered_alerts

    def _create_alert(self, alert_type, message, priority):
        """Create an alert object"""
        return {
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        }

    def _check_abandoned_objects(self, objects):
        """Check for potentially abandoned objects"""
        # This would typically compare with previous frames
        # For now, we'll use a simple heuristic
        stationary_objects = ['backpack', 'bag', 'luggage', 'package']
        stationary_count = len([obj for obj in objects if any(so in obj['class'] for so in stationary_objects)])
        
        return stationary_count > 2

    def _filter_alerts_by_cooldown(self, new_alerts):
        """Filter alerts to avoid duplicates within cooldown period"""
        current_time = datetime.now()
        valid_alerts = []
        
        for alert in new_alerts:
            # Check if similar alert was recently sent
            recent_similar = any(
                existing_alert['type'] == alert['type'] and 
                (current_time - datetime.fromisoformat(existing_alert['timestamp'])) < timedelta(seconds=self.alert_cooldown)
                for existing_alert in self.alert_history
            )
            
            if not recent_similar:
                valid_alerts.append(alert)
                self.alert_history.append(alert)
                
                # Clean old alerts from history
                self.alert_history = [
                    a for a in self.alert_history 
                    if (current_time - datetime.fromisoformat(a['timestamp'])) < timedelta(minutes=10)
                ]
        
        return valid_alerts
