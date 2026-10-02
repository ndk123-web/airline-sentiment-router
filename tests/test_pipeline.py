"""
AirRoute AI - Unit & Integration Test Suite
Tests data processing, clause segmentation, department routing, ticket DB CRUD, and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.routing.router import AirlineReviewRouter, router
from src.tickets.ticket_manager import (
    create_ticket_from_issue,
    list_tickets,
    update_ticket_status,
    get_ticket_statistics,
    process_review_and_create_tickets
)


@pytest.fixture
def client():
    return TestClient(app)


def test_router_sentiment_prediction():
    """Test model sentiment prediction output format."""
    res = router.predict_overall_sentiment("Flight was amazing, best service ever!")
    assert "sentiment" in res
    assert "confidence" in res
    assert res["sentiment"] in ["positive", "neutral", "negative"]


def test_clause_segmentation():
    """Test compound review splitting into distinct clauses."""
    compound_text = "Flight was delayed 3 hours, luggage was lost in Denver, but flight attendants were very kind."
    clauses = router.split_into_clauses(compound_text)
    assert len(clauses) >= 2
    assert any("delayed" in c.lower() for c in clauses)
    assert any("luggage" in c.lower() or "lost" in c.lower() for c in clauses)


def test_department_routing_logic():
    """Test routing mapping to specific operational departments."""
    res = router.analyze_and_route(
        "@united My flight was delayed by four hours and you lost my baggage in Chicago, but the cabin crew was helpful!",
        airline_name="United"
    )
    assert res["total_issues_detected"] >= 2
    assert "Flight Operations" in res["target_departments"]
    assert "Baggage Services" in res["target_departments"]


def test_ticket_database_crud():
    """Test creating, listing, updating, and querying tickets."""
    test_review = "@Delta Terrible experience, my flight was cancelled without refund!"
    result = process_review_and_create_tickets(test_review, airline_name="Delta", actionable_only=True)
    
    assert "generated_tickets" in result
    assert len(result["generated_tickets"]) > 0
    tkt_id = result["generated_tickets"][0]["ticket_id"]

    # Test update
    updated = update_ticket_status(tkt_id, "Resolved", resolution_notes="Full refund processed.")
    assert updated is not None
    assert updated["status"] == "Resolved"
    assert updated["resolution_notes"] == "Full refund processed."


def test_fastapi_endpoints(client):
    """Test FastAPI REST endpoints."""
    # Root
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.json()["system"] == "AirRoute AI"

    # Predict
    predict_res = client.post("/api/predict", json={
        "text": "@AmericanAir Lost my suitcase on flight 202!",
        "airline": "American",
        "create_tickets": False
    })
    assert predict_res.status_code == 200
    data = predict_res.json()
    assert "overall_sentiment" in data
    assert "extracted_issues" in data

    # Ticket stats
    stats_res = client.get("/api/analytics/tickets")
    assert stats_res.status_code == 200
    assert "total_tickets" in stats_res.json()
