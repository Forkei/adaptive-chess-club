from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.models.base import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)          # 1–5, optional
    username = Column(String(64), nullable=True)     # logged-in username, or null
    page_url = Column(String(512), nullable=True)    # Referer header
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
