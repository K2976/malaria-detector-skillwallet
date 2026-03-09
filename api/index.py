import os
import sys

# Add project root to sys.path so web_app and src can be found
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app.app import app

# Vercel needs "app" exposed
