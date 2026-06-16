from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from .database import Base

class Analysis(Base):
    __tablename__ = 'analyses'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    attack_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    anomaly = Column(Boolean, default=False)
    total_records = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
