"""Compatibility exports; new code should use :mod:`config.settings`."""

from .settings import DatabaseSettings


_settings = DatabaseSettings.from_env()
DB_HOST = _settings.host
DB_PORT = _settings.port
DB_NAME = _settings.database
DB_USER = _settings.user
DB_PASSWORD = _settings.password
SQLALCHEMY_URL = _settings.sqlalchemy_url
