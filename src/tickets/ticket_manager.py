"""
AirRoute AI - Ticket Management System
Handles persistent SQLite storage, lifecycle state machine (Open -> In Progress -> Resolved -> Closed),
and query filtering for actionable passenger complaint tickets.
"""

import os
import uuid
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.routing.router import router

DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "tickets.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class SupportTicket(Base):
    """
    SQLAlchemy Model for an Airline Passenger Support Ticket.
    """
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(32), unique=True, index=True, nullable=False)
    airline = Column(String(64), default="Unknown", index=True)
    department = Column(String(64), index=True, nullable=False)
    priority = Column(String(16), index=True, default="MEDIUM")
    status = Column(String(24), index=True, default="Open")
    sentiment = Column(String(16), default="negative")
    issue_description = Column(Text, nullable=False)
    original_review = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    resolution_notes = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "airline": self.airline,
            "department": self.department,
            "priority": self.priority,
            "status": self.status,
            "sentiment": self.sentiment,
            "issue_description": self.issue_description,
            "original_review": self.original_review,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
            "resolution_notes": self.resolution_notes or ""
        }


def init_db():
    """Initializes SQLite database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Yields a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def generate_ticket_id() -> str:
    """Generates a human-friendly ticket ID: TKT-YYYYMMDD-XXXX"""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:5].upper()
    return f"TKT-{today}-{short_uuid}"


def create_ticket_from_issue(
    original_review: str,
    issue_description: str,
    department: str,
    priority: str = "MEDIUM",
    sentiment: str = "negative",
    airline: str = "Unknown",
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Creates and persists a single support ticket.
    """
    session = db or SessionLocal()
    try:
        ticket = SupportTicket(
            ticket_id=generate_ticket_id(),
            airline=airline,
            department=department,
            priority=priority,
            status="Open",
            sentiment=sentiment,
            issue_description=issue_description,
            original_review=original_review,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            updated_at=datetime.datetime.now(datetime.timezone.utc)
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket.to_dict()
    finally:
        if not db:
            session.close()


def process_review_and_create_tickets(
    review_text: str,
    airline_name: str = "Unknown",
    actionable_only: bool = True
) -> Dict[str, Any]:
    """
    Analyzes a review with the router and creates support tickets for actionable issues.
    """
    analysis = router.analyze_and_route(review_text, airline_name=airline_name)
    created_tickets = []
    
    session = SessionLocal()
    try:
        for issue in analysis.get("extracted_issues", []):
            if actionable_only and not issue.get("is_actionable", False):
                continue
            
            tkt = SupportTicket(
                ticket_id=generate_ticket_id(),
                airline=airline_name,
                department=issue["department"],
                priority=issue["priority"],
                status="Open",
                sentiment=issue["sentiment"],
                issue_description=issue["issue_description"],
                original_review=review_text,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                updated_at=datetime.datetime.now(datetime.timezone.utc)
            )
            session.add(tkt)
            session.commit()
            session.refresh(tkt)
            created_tickets.append(tkt.to_dict())
    finally:
        session.close()

    analysis["generated_tickets"] = created_tickets
    return analysis


def list_tickets(
    department: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    airline: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Lists support tickets with optional filtering.
    """
    session = SessionLocal()
    try:
        query = session.query(SupportTicket)
        if department and department != "All":
            query = query.filter(SupportTicket.department == department)
        if status and status != "All":
            query = query.filter(SupportTicket.status == status)
        if priority and priority != "All":
            query = query.filter(SupportTicket.priority == priority)
        if airline and airline != "All":
            query = query.filter(SupportTicket.airline == airline)

        tickets = query.order_by(desc(SupportTicket.created_at)).limit(limit).all()
        return [t.to_dict() for t in tickets]
    finally:
        session.close()


def update_ticket_status(
    ticket_id: str,
    new_status: str,
    resolution_notes: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Updates the status and optional resolution notes for a ticket.
    Valid statuses: 'Open', 'In Progress', 'Resolved', 'Closed'
    """
    session = SessionLocal()
    try:
        ticket = session.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id).first()
        if not ticket:
            return None

        ticket.status = new_status
        ticket.updated_at = datetime.datetime.now(datetime.timezone.utc)
        if resolution_notes is not None:
            ticket.resolution_notes = resolution_notes

        session.commit()
        session.refresh(ticket)
        return ticket.to_dict()
    finally:
        session.close()


def delete_ticket(ticket_id: str) -> bool:
    """
    Deletes a single support ticket by its ticket_id.
    """
    session = SessionLocal()
    try:
        ticket = session.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id).first()
        if not ticket:
            return False
        session.delete(ticket)
        session.commit()
        return True
    finally:
        session.close()


def delete_all_tickets() -> int:
    """
    Deletes all support tickets from the database. Returns count of deleted records.
    """
    session = SessionLocal()
    try:
        deleted_count = session.query(SupportTicket).delete()
        session.commit()
        return deleted_count
    finally:
        session.close()


def get_ticket_statistics() -> Dict[str, Any]:
    """
    Computes ticket dashboard statistics (status counts, department distribution, priority levels).
    """
    session = SessionLocal()
    try:
        all_tickets = session.query(SupportTicket).all()
        total = len(all_tickets)

        status_counts = {"Open": 0, "In Progress": 0, "Resolved": 0, "Closed": 0}
        priority_counts = {"URGENT": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        dept_counts = {}

        for t in all_tickets:
            status_counts[t.status] = status_counts.get(t.status, 0) + 1
            priority_counts[t.priority] = priority_counts.get(t.priority, 0) + 1
            dept_counts[t.department] = dept_counts.get(t.department, 0) + 1

        return {
            "total_tickets": total,
            "status_breakdown": status_counts,
            "priority_breakdown": priority_counts,
            "department_breakdown": dept_counts,
            "open_tickets": status_counts.get("Open", 0) + status_counts.get("In Progress", 0),
            "resolved_tickets": status_counts.get("Resolved", 0) + status_counts.get("Closed", 0)
        }
    finally:
        session.close()


# Initialize database on module load
init_db()


if __name__ == "__main__":
    stats = get_ticket_statistics()
    print("\n--- TICKET MANAGEMENT SYSTEM INITIALIZED ---")
    print("Ticket Stats:", stats)
