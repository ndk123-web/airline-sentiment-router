"""
AirRoute AI - Ticket & Multi-Tenant User Management System
Handles persistent SQLite storage, lifecycle state machine (Open -> In Progress -> Resolved -> Closed),
user authentication (Super Admin & Airline Tenants), and query filtering for actionable passenger complaint tickets.
"""

import os
import uuid
import hashlib
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


# -------------------------------------------------------------
# User Model (Multi-Tenant Authentication)
# -------------------------------------------------------------

class User(Base):
    """
    SQLAlchemy Model for Users (Super Admin & Airline Managers).
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), default="airline_admin", nullable=False)  # "super_admin" or "airline_admin"
    airline = Column(String(64), default="All", nullable=False)        # "All" or e.g. "United", "Delta"
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "airline": self.airline,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None
        }


# -------------------------------------------------------------
# Support Ticket Model
# -------------------------------------------------------------

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
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))
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


# -------------------------------------------------------------
# Password Utilities
# -------------------------------------------------------------

def hash_password(password: str, salt: str = "airroute_salt_2026") -> str:
    """Hashes a password with salt using SHA-256."""
    return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str, salt: str = "airroute_salt_2026") -> bool:
    """Verifies a plain password against the stored hash."""
    return hash_password(password, salt) == password_hash


# -------------------------------------------------------------
# User CRUD & Authentication Functions
# -------------------------------------------------------------

def register_user(
    username: str,
    email: str,
    password: str,
    airline: str = "United",
    role: str = "airline_admin"
) -> Dict[str, Any]:
    """
    Registers a new user (airline account or admin).
    Returns {"success": bool, "user": dict, "error": str}.
    """
    session = SessionLocal()
    try:
        existing_user = session.query(User).filter(
            (User.username == username.strip().lower()) | (User.email == email.strip().lower())
        ).first()

        if existing_user:
            return {"success": False, "error": "Username or email is already registered."}

        new_user = User(
            username=username.strip().lower(),
            email=email.strip().lower(),
            password_hash=hash_password(password),
            role=role,
            airline=airline.strip(),
            created_at=datetime.datetime.now(datetime.timezone.utc)
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return {"success": True, "user": new_user.to_dict()}
    finally:
        session.close()


def authenticate_user(username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user with username/email and password.
    Returns user dict on success, None on failure.
    """
    session = SessionLocal()
    try:
        clean_identifier = username_or_email.strip().lower()
        user = session.query(User).filter(
            (User.username == clean_identifier) | (User.email == clean_identifier)
        ).first()

        if not user:
            return None

        if verify_password(password, user.password_hash):
            return user.to_dict()
        return None
    finally:
        session.close()


def list_users() -> List[Dict[str, Any]]:
    """Returns a list of all registered users."""
    session = SessionLocal()
    try:
        users = session.query(User).all()
        return [u.to_dict() for u in users]
    finally:
        session.close()


def seed_default_users():
    """Seeds default admin and sample airline accounts if not already present."""
    session = SessionLocal()
    try:
        default_accounts = [
            ("admin", "admin@airroute.ai", "admin123", "super_admin", "All"),
            ("united", "ops@united.com", "united123", "airline_admin", "United"),
            ("delta", "ops@delta.com", "delta123", "airline_admin", "Delta"),
            ("american", "ops@aa.com", "american123", "airline_admin", "American"),
            ("southwest", "ops@southwest.com", "southwest123", "airline_admin", "Southwest"),
        ]

        for username, email, pwd, role, airline in default_accounts:
            existing = session.query(User).filter(User.username == username).first()
            if not existing:
                u = User(
                    username=username,
                    email=email,
                    password_hash=hash_password(pwd),
                    role=role,
                    airline=airline,
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                session.add(u)
        session.commit()
    finally:
        session.close()


# -------------------------------------------------------------
# Ticket Core Functions
# -------------------------------------------------------------

def init_db():
    """Initializes SQLite database tables and seeds default users."""
    Base.metadata.create_all(bind=engine)
    seed_default_users()


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
    Creates and persists a support ticket from an identified operational issue.
    """
    session = db or SessionLocal()
    close_session = db is None

    try:
        ticket = SupportTicket(
            ticket_id=generate_ticket_id(),
            airline=airline,
            department=department,
            priority=priority.upper(),
            status="Open",
            sentiment=sentiment.lower(),
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
        if close_session:
            session.close()


def process_review_and_create_tickets(
    review_text: str,
    airline_name: str = "Unknown",
    actionable_only: bool = True
) -> Dict[str, Any]:
    """
    End-to-End Workflow:
    1. Routes review through Clause Splitter + Sentiment ML + NLI Zero-Shot Router.
    2. Filters for actionable complaints (negative operational clauses).
    3. Persists generated tickets in SQLite.
    4. Returns analysis enriched with persisted ticket metadata.
    """
    analysis = router.analyze_and_route(review_text=review_text, airline_name=airline_name)

    created_tickets = []
    session = SessionLocal()
    try:
        issues_list = analysis.get("extracted_issues") or analysis.get("detected_issues", [])
        for issue in issues_list:
            if actionable_only and not issue.get("is_actionable", False):
                continue

            tkt = SupportTicket(
                ticket_id=generate_ticket_id(),
                airline=airline_name or "Unknown",
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


def delete_all_tickets(airline: Optional[str] = None) -> int:
    """
    Deletes support tickets from the database.
    If airline is specified, deletes only tickets for that airline.
    Returns count of deleted records.
    """
    session = SessionLocal()
    try:
        query = session.query(SupportTicket)
        if airline and airline != "All":
            query = query.filter(SupportTicket.airline == airline)
        deleted_count = query.delete(synchronize_session=False)
        session.commit()
        return deleted_count
    finally:
        session.close()


def get_ticket_statistics(airline: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes ticket dashboard statistics (status counts, department distribution, priority levels).
    Optionally filters by airline for multi-tenant isolation.
    """
    session = SessionLocal()
    try:
        query = session.query(SupportTicket)
        if airline and airline != "All":
            query = query.filter(SupportTicket.airline == airline)

        all_tickets = query.all()
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


# Initialize database and default users on module load
init_db()


if __name__ == "__main__":
    stats = get_ticket_statistics()
    users = list_users()
    print("\n--- TICKET MANAGEMENT SYSTEM & AUTH INITIALIZED ---")
    print(f"Users seeded: {len(users)}")
    print("Ticket Stats:", stats)
