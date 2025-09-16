#!/usr/bin/env python3
"""
Medical Analytics Platform - Main Entry Point
"""

import sys
import os

# Add project root and src directory to Python path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from medical_analytics.core.app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)