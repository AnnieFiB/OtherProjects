from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, ARRAY, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database.database import Base

class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    local_govt = Column(String, index=True)
    town = Column(String)
    street_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    electricity = relationship("Electricity", back_populates="location", uselist=False)
    flood = relationship("Flood", back_populates="location", uselist=False)
    road = relationship("Road", back_populates="location", uselist=False)
    waste = relationship("Waste", back_populates="location", uselist=False)
    security = relationship("Security", back_populates="location", uselist=False)
    water = relationship("Water", back_populates="location", uselist=False)
    medical = relationship("Medical", back_populates="location", uselist=False)
    noise = relationship("Noise", back_populates="location", uselist=False)

class Electricity(Base):
    __tablename__ = "electricity"

    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, ForeignKey("locations.id"))
    hours_of_power_per_day = Column(String)
    days_with_electricity_per_week = Column(String)
    outage_reasons = Column(ARRAY(String))
    unplanned_outage_frequency = Column(String)
    outage_times = Column(ARRAY(String))
    power_restoration_time = Column(String)
    voltage_fluctuation = Column(String)
    backup_power_sources = Column(ARRAY(String))
    electricity_reliability = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="electricity")

class Flood(Base):
    __tablename__ = "flood"

    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, ForeignKey("locations.id"))
    rainfall_frequency = Column(String)
    flood_likelihood = Column(String)
    flood_water_dissipation_time = Column(String)
    flood_severity = Column(String)
    drainage_blockage_frequency = Column(String)
    flood_causes = Column(ARRAY(String))
    flood_risk_sentiment = Column(String)
    flood_prevention_measures = Column(ARRAY(String))
    drainage_system_rating = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="flood")

