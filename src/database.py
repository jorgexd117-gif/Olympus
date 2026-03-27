from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
import os
import re


def _make_async_url(url: str) -> str:
    url = url.replace("postgresql://", "postgresql+asyncpg://").replace(
        "postgres://", "postgresql+asyncpg://"
    )
    url = re.sub(r"[?&]sslmode=[^&]*", "", url)
    return url


engine = create_async_engine(
    _make_async_url(os.getenv("DATABASE_URL", "")),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass
