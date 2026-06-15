from routes.dashboard_routes import dashboard_bp
from routes.packer_routes import packer_bp
from routes.monitoring_routes import monitoring_bp
from routes.video_processing_routes import video_bp
from routes.reports_routes import reports_bp

__all__ = [
    'dashboard_bp',
    'packer_bp',
    'monitoring_bp',
    'video_bp',
    'reports_bp'
]