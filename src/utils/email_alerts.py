import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv  
from pathlib import Path

# Force load the .env file from the project root
# This looks for .env in the current folder OR parent folders
load_dotenv() 

class EmailAlerter:
    """Simple email alerting system"""
    
    def __init__(self):
        # Debug print to help us see what's happening
        print(f"   [DEBUG] Loading credentials... Email: {os.getenv('ALERT_EMAIL')}")
        
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('ALERT_EMAIL')
        self.sender_password = os.getenv('ALERT_EMAIL_PASSWORD')
        self.recipient_email = os.getenv('ALERT_RECIPIENT_EMAIL')
        
        # Check if credentials exist
        self.enabled = all([self.sender_email, self.sender_password, self.recipient_email])
        
        if not self.enabled:
            print("Email alerts disabled (credentials not configured or missing)")
            print("Make sure .env file is in the project root and variables are named correctly.")
    
    def send_alert(self, subject: str, body: str):
        """Send email alert"""
        if not self.enabled:
            print(f"[ALERT - EMAIL DISABLED] {subject}")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"[Churn Model Alert] {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"Email sent successfully: {subject}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    # ... (Keep the rest of your methods: alert_performance_drop, alert_drift_detected, etc.) ...
    
    def alert_drift_detected(self, drift_info: Dict[str, Any]):
        """Alert when drift is detected"""
        subject = "Data Drift Detected"
        body = f"""
Significant data drift detected:

Features drifted: {drift_info['features_drifted']}/{drift_info['features_checked']}
Drift rate: {drift_info['drift_rate']:.1%}
Top drifted features: {', '.join(drift_info['drifted_features'][:5])}

Timestamp: {datetime.now().isoformat()}

Action required: The system has triggered retraining automatically.
        """
        self.send_alert(subject, body)
    
    def alert_retraining_started(self, reason: str):
        """Alert when retraining starts"""
        subject = "Model Retraining Started"
        body = f"""
Automated model retraining has been triggered.

Reason: {reason}
Timestamp: {datetime.now().isoformat()}

The system will automatically deploy if the new model performs better.
        """
        self.send_alert(subject, body)
    
    def alert_retraining_completed(self, summary: Dict[str, Any]):
        """Alert when retraining completes"""
        deployed = "DEPLOYED" if summary['deployed'] else "✗ NOT DEPLOYED"
        subject = f"Model Retraining Complete - {deployed}"
        
        # Safe access to dict keys
        best_model = summary.get('best_model', 'Unknown')
        best_score = summary.get('best_score', 0.0)
        reason = summary.get('comparison_reason', 'N/A')
        duration = summary.get('duration_seconds', 0)
        timestamp = summary.get('timestamp', str(datetime.now()))

        body = f"""
Model retraining has completed.

Best model: {best_model}
Best score: {best_score:.4f}
Deployed: {summary['deployed']}
Reason: {reason}

Duration: {duration:.1f} seconds
Timestamp: {timestamp}
        """
        self.send_alert(subject, body)

# Convenience function
def get_alerter():
    """Get configured email alerter instance"""
    return EmailAlerter()