"""Gunicorn and local Flask entry point."""

from persistency.web import create_app

app = create_app()
