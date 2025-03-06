# This script contains example model definitions for the SQLAlchemy models.
# Modify the script to add or remove model definitions as needed.
# The script is executed by the setup.py module to generate the model definitions in the models directory.
# The Tests that are written in the tests/test_models.py file are based on these model definitions.

MODEL_DEFINITIONS_SCRIPT = {
    "models/__init__.py": '''from .base import Base
from .user import User
from .user_post import UserPost
''',

    "models/base.py": '''from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData, Column

# Define naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)

class Base:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"

Base = declarative_base(cls=Base, metadata=metadata)''',

    "models/user.py": '''from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, UniqueConstraint, Index, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class User(Base):
    __tablename__ = "user"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    username = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationship with UserPost
    posts = relationship("UserPost", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        PrimaryKeyConstraint('id', name='pk_user'),
        UniqueConstraint('username', name='uq_user_username'),
        UniqueConstraint('email', name='uq_user_email'),
        Index('ix_user_id', 'id', unique=True),
        Index('ix_user_username', 'username'),
        {'schema': None}
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', username='{self.username}')>"''',

    "models/user_post.py": '''from sqlalchemy import Column, Boolean, String, Text, DateTime, ForeignKey, Index, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class UserPost(Base):
    __tablename__ = "user_post"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True) # for soft delete
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationship with User
    user = relationship("User", back_populates="posts")
    
    __table_args__ = (
        PrimaryKeyConstraint('id', name='pk_user_post'),
        Index('ix_user_post_id', 'id', unique=True),
        Index('ix_user_post_user_id', 'user_id'),
        {'schema': None}
    )
    
    def __repr__(self):
        return f"<UserPost(id={self.id}, user_id={self.user_id}, title='{self.title}')>"''',
}