"""
Vercel Serverless Handler for Flask App
This file adapts the Flask app for Vercel's serverless environment
"""

from app import app

def handler(request, response):
    return app(request, response)


application = app
