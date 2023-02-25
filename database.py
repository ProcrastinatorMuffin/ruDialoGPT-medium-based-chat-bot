from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, MetaData, Table
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

engine = create_engine('sqlite:///data.db', echo=True)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer)
    username = Column(String)
    info = relationship('UserInfo', backref='user', uselist=False)

    def __init__(self, chat_id, username):
        self.chat_id = chat_id
        self.username = username

class UserInfo(Base):
    __tablename__ = 'specific_user_info'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey('users.chat_id'))
    name = Column(String)
    age = Column(Integer)
    country = Column(String)
    interests = Column(String)

    def __init__(self, chat_id, name, age, country, interests):
        self.chat_id = chat_id
        self.name = name
        self.age = age
        self.country = country
        self.interests = interests

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()