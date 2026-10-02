"""
AirRoute AI - FastAPI REST API Service
Provides RESTful endpoints for real-time review sentiment prediction,
multi-issue intelligent department routing, support ticket management,
and Big Data analytics & model evaluation metrics.
"""

import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.routing.router import router
from src.tickets.ticket_manager import (
    list_tickets,
    get_ticket_statistics,
    update_ticket_status,
    process_review_and_create_tickets,
    create_ticket_from_issue
)

app = FastAPI(
    title="AirRoute AI - API",
    description="Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Pydantic Request / Response Schemas
# -------------------------------------------------------------

class ReviewAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=3, example="@united My flight was delayed by 4 hours and you lost my luggage!")
    airline: Optional[str] = Field("Unknown", example="United")
    create_tickets: Optional[bool] = Field(False, description="Automatically persist actionable tickets to DB")


class TicketStatusUpdateRequest(BaseModel):
    status: str = Field(..., example="Resolved")
    resolution_notes: Optional[str] = Field(None, example="Refund of $150 issued and luggage tracked to baggage carousel 4.")


class ManualTicketCreateRequest(BaseModel):
    original_review: str
    issue_description: str
    department: str
    priority: Optional[str] = "MEDIUM"
    sentiment: Optional[str] = "negative"
    airline: Optional[str] = "Unknown"


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "system": "AirRoute AI",
        "description": "Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System",
        "status": "online",
        "docs_url": "/docs"
    }


@app.post("/api/predict", tags=["NLP & Routing"])
def predict_and_route(request: ReviewAnalysisRequest):
    """
    Analyzes a passenger review:
    1. Evaluates overall sentiment polarity and confidence score.
    2. Segments review into grammatical sub-clauses.
    3. Identifies operational issue categories and matches to 6 airline departments.
    4. Computes priority levels (URGENT, HIGH, MEDIUM, LOW).
    5. Optionally creates actionable support tickets in the database.
    """
    if request.create_tickets:
        result = process_review_and_create_tickets(
            review_text=request.text,
            airline_name=request.airline or "Unknown",
            actionable_only=True
        )
    else:
        result = router.analyze_and_route(
            review_text=request.text,
            airline_name=request.airline or "Unknown"
        )
    return result


@app.get("/api/tickets", tags=["Ticket Management"])
def get_tickets(
    department: Optional[str] = Query(None, description="Filter by department"),
    status: Optional[str] = Query(None, description="Filter by status (Open, In Progress, Resolved, Closed)"),
    priority: Optional[str] = Query(None, description="Filter by priority (URGENT, HIGH, MEDIUM, LOW)"),
    airline: Optional[str] = Query(None, description="Filter by airline"),
    limit: int = Query(100, ge=1, le=500)
):
    """
    Retrieves filtered list of passenger support tickets.
    """
    tickets = list_tickets(
        department=department,
        status=status,
        priority=priority,
        airline=airline,
        limit=limit
    )
    return {"total": len(tickets), "tickets": tickets}


@app.patch("/api/tickets/{ticket_id}", tags=["Ticket Management"])
def patch_ticket_status(ticket_id: str, request: TicketStatusUpdateRequest):
    """
    Updates ticket status and adds resolution notes.
    Valid statuses: 'Open', 'In Progress', 'Resolved', 'Closed'
    """
    valid_statuses = ["Open", "In Progress", "Resolved", "Closed"]
    if request.status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{request.status}'. Allowed values: {valid_statuses}"
        )

    updated = update_ticket_status(
        ticket_id=ticket_id,
        new_status=request.status,
        resolution_notes=request.resolution_notes
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")
    return {"message": "Ticket updated successfully", "ticket": updated}


@app.post("/api/tickets/create", tags=["Ticket Management"])
def create_manual_ticket(request: ManualTicketCreateRequest):
    """
    Manually creates a support ticket.
    """
    tkt = create_ticket_from_issue(
        original_review=request.original_review,
        issue_description=request.issue_description,
        department=request.department,
        priority=request.priority or "MEDIUM",
        sentiment=request.sentiment or "negative",
        airline=request.airline or "Unknown"
    )
    return {"message": "Ticket created successfully", "ticket": tkt}


@app.get("/api/analytics/tickets", tags=["Analytics"])
def get_ticket_analytics():
    """
    Returns aggregated metrics on support tickets (status counts, priority breakdown, department distribution).
    """
    return get_ticket_statistics()


@app.get("/api/models/evaluation", tags=["Analytics & Scalability"])
def get_model_evaluation_metrics():
    """
    Returns comparative evaluation metrics: Baseline (Scikit-Learn) vs Distributed MLlib (PySpark).
    """
    summary_path = "results/model_comparison_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)

    # Fallback to individual metrics
    res = {}
    if os.path.exists("results/baseline_metrics.json"):
        with open("results/baseline_metrics.json", "r") as f:
            res["baseline"] = json.load(f)
    if os.path.exists("results/spark_mllib_metrics.json"):
        with open("results/spark_mllib_metrics.json", "r") as f:
            res["spark_mllib"] = json.load(f)
    return res
