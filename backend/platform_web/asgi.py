import os
import sys
from pathlib import Path
from django.core.asgi import get_asgi_application

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "platform_web.settings")

application = get_asgi_application()
