import os

from decouple import config
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", 'ai_chatbot.settings')
app = Celery("ai_chatbot")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

