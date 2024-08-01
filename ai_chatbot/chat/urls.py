# chat/urls.py

from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from .views import chat_view, upload_multiple_docs, add_to_database, upload_file, trigger_fine_tuning

app_name = "chat"

urlpatterns = [path("", chat_view, name="chat_view"), 
path("upload_multiple_docs", upload_multiple_docs, name="upload_to_multiple_docs"),
path("add_to_database", add_to_database, name="upload_to_db"),
path("upload_file", upload_file, name='upload_csv'),
path("trigger_fine_tuning", trigger_fine_tuning, name='trigger_fine_tuning')] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
