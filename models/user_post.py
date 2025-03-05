from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base

class UserPost(Base):
    __tablename__ = "user_post"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
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
        return f"<UserPost(id={self.id}, user_id={self.user_id}, title='{self.title}')>"