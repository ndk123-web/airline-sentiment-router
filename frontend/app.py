"""
AirRoute AI Enterprise - Intelligent Airline Passenger Sentiment & Review Routing Platform
Multi-Tenant SaaS Portal for Airlines and System Administrators.
Professional Operational Design tailored for Enterprise Aviation Management.
"""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Direct local modules
from src.routing.router import router
from src.tickets.ticket_manager import (
    list_tickets,
    get_ticket_statistics,
    update_ticket_status,
    delete_ticket,
    delete_all_tickets,
    process_review_and_create_tickets,
    authenticate_user,
    register_user,
    list_users
)

FASTAPI_BASE_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

# Set page configuration
st.set_page_config(
    page_title="AirRoute AI Enterprise | Operational Feedback & Triage Desk",
    page_icon="AIR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean Corporate Theme with Psychological Hierarchy)
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Primary Headings with Requested Brand Blue */
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1D4ED8 !important; /* Corporate Blue */
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }

    /* Executive Alert Banner */
    .executive-alert {
        background: #FEF2F2;
        border-left: 4px solid #DC2626;
        border-radius: 6px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 1.25rem;
        color: #991B1B;
        font-size: 0.9rem;
    }
    .executive-alert strong {
        color: #7F1D1D;
        font-weight: 700;
    }

    /* Operational Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 1.15rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.75rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
    }

    /* Priority Badges */
    .badge-urgent {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid #FECACA;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid #FED7AA;
    }
    .badge-medium {
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid #DBEAFE;
    }
    .badge-low {
        background-color: #F0FDF4;
        color: #166534;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid #DCFCE7;
    }

    /* Triage Result Box */
    .triage-box {
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1D4ED8;
        background: #F8FAFC;
    }

    /* Clean Sidebar Branding without Emojis */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sidebar-tenant-card {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 12px;
    }
    .sidebar-tenant-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-tenant-name {
        font-size: 1.05rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .sidebar-tenant-user {
        font-size: 0.8rem;
        color: #475569;
    }

    /* Table & Container Polish */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None


# Check FastAPI Backend Connectivity
def check_api_health() -> bool:
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/", timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False

API_ONLINE = check_api_health()


# -------------------------------------------------------------
# API Client Helpers
# -------------------------------------------------------------
def api_predict_and_route(text: str, airline: str, create_tickets: bool = True) -> dict:
    if API_ONLINE:
        try:
            resp = requests.post(
                f"{FASTAPI_BASE_URL}/api/predict",
                json={"text": text, "airline": airline, "create_tickets": create_tickets},
                timeout=4.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    if create_tickets:
        return process_review_and_create_tickets(text, airline_name=airline, actionable_only=True)
    return router.analyze_and_route(text, airline_name=airline)


def api_list_tickets(department=None, status=None, priority=None, airline=None, limit=200):
    if API_ONLINE:
        try:
            params = {"limit": limit}
            if department and department != "All":
                params["department"] = department
            if status and status != "All":
                params["status"] = status
            if priority and priority != "All":
                params["priority"] = priority
            if airline and airline != "All":
                params["airline"] = airline
            resp = requests.get(f"{FASTAPI_BASE_URL}/api/tickets", params=params, timeout=3.0)
            if resp.status_code == 200:
                return resp.json().get("tickets", [])
        except Exception:
            pass
    return list_tickets(department=department, status=status, priority=priority, airline=airline, limit=limit)


def api_update_ticket_status(ticket_id: str, new_status: str, notes: str = None):
    if API_ONLINE:
        try:
            resp = requests.patch(
                f"{FASTAPI_BASE_URL}/api/tickets/{ticket_id}",
                json={"status": new_status, "resolution_notes": notes},
                timeout=3.0
            )
            if resp.status_code == 200:
                return resp.json().get("ticket")
        except Exception:
            pass
    return update_ticket_status(ticket_id, new_status, resolution_notes=notes)


def api_delete_ticket(ticket_id: str):
    if API_ONLINE:
        try:
            resp = requests.delete(f"{FASTAPI_BASE_URL}/api/tickets/{ticket_id}", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
    return delete_ticket(ticket_id)


def api_delete_all_tickets(airline: str = None):
    if API_ONLINE:
        try:
            params = {"airline": airline} if airline else {}
            resp = requests.delete(f"{FASTAPI_BASE_URL}/api/tickets", params=params, timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
    return delete_all_tickets(airline=airline) > 0


# -------------------------------------------------------------
# Data Loaders
# -------------------------------------------------------------
@st.cache_data
def load_processed_data():
    parquet_path = "data/processed/airline_reviews.parquet"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    csv_path = "data/raw/Tweets.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ==============================================================================
# AUTHENTICATION SCREEN (LOGIN / SIGNUP - PROFESSIONAL & CLEAN)
# ==============================================================================
if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #1D4ED8; letter-spacing: -0.03em; margin-bottom: 0.35rem;">
            AirRoute AI Enterprise
        </h1>
        <p style="font-size: 1.0rem; color: #64748B; font-weight: 500;">
            Aviation Feedback Analytics & Real-Time Incident Triage Desk
        </p>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 1.4, 1])

    with auth_col2:
        tab_login, tab_signup = st.tabs(["Sign In to Airline Portal", "Register Airline Account"])

        # ----------------- LOGIN TAB -----------------
        with tab_login:
            st.markdown("##### Portal Authentication")
            login_identity = st.text_input("Username or Corporate Email", placeholder="e.g. united or ops@united.com", key="login_id")
            login_password = st.text_input("Password", type="password", placeholder="Enter your credentials", key="login_pwd")

            if st.button("Sign In", type="primary", use_container_width=True):
                if not login_identity or not login_password:
                    st.error("Please provide both username/email and password.")
                else:
                    user = authenticate_user(login_identity, login_password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.success(f"Authenticated as {user['username'].upper()}. Launching portal...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please verify your username and password.")

            st.markdown("---")
            st.caption("Standard Accounts: `united` / `united123` (United), `delta` / `delta123` (Delta), `admin` / `admin123` (Super Admin).")

        # ----------------- SIGNUP TAB -----------------
        with tab_signup:
            st.markdown("##### New Airline Registration")
            new_username = st.text_input("Desired Username", placeholder="e.g. emirates_ops", key="reg_user")
            new_email = st.text_input("Corporate Work Email", placeholder="e.g. ops@emirates.com", key="reg_email")
            new_airline = st.selectbox(
                "Select Airline Carrier",
                ["United", "Delta", "American", "Southwest", "US Airways", "Virgin America", "JetBlue", "Alaska Airlines", "Other"],
                key="reg_airline"
            )
            custom_airline = ""
            if new_airline == "Other":
                custom_airline = st.text_input("Specify Airline Name:", placeholder="e.g. Emirates", key="reg_custom_airline")

            new_password = st.text_input("Access Password", type="password", placeholder="Minimum 4 characters", key="reg_pwd")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pwd_confirm")

            if st.button("Register Airline Tenant", type="primary", use_container_width=True):
                final_airline = custom_airline.strip() if new_airline == "Other" else new_airline
                if not new_username or not new_email or not new_password:
                    st.error("All fields are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    reg_res = register_user(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        airline=final_airline,
                        role="airline_admin"
                    )
                    if reg_res.get("success"):
                        st.success(f"Tenant '{final_airline}' registered successfully. Please sign in.")
                    else:
                        st.error(reg_res.get("error", "Registration error."))

    st.stop()


# ==============================================================================
# AUTHENTICATED APPLICATION
# ==============================================================================
current_user = st.session_state["user"]
is_super_admin = current_user.get("role") == "super_admin"
user_airline = current_user.get("airline", "United")

# Load Dataset
raw_df = load_processed_data()
airline_col = "airline_clean" if "airline_clean" in raw_df.columns else "airline"
available_airlines = sorted(raw_df[airline_col].dropna().unique().tolist()) if not raw_df.empty else ["United", "Delta", "American", "Southwest", "US Airways", "Virgin America"]

# ----------------- SIDEBAR CONTROLS (STRICTLY NO EMOJIS) -----------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 0.85rem;">
        <div class="sidebar-header">AirRoute AI</div>
        <div style="font-size: 0.8rem; color: #64748B; font-weight: 500;">Enterprise Operations Desk</div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Tenant Profile Card
    if is_super_admin:
        st.markdown(f"""
        <div class="sidebar-tenant-card" style="border-left: 3px solid #D97706;">
            <div class="sidebar-tenant-label" style="color: #D97706;">Access Level: Super Admin</div>
            <div class="sidebar-tenant-name" style="color: #B45309;">Global Portal</div>
            <div class="sidebar-tenant-user">Operator: {current_user['username']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sidebar-tenant-card" style="border-left: 3px solid #1D4ED8;">
            <div class="sidebar-tenant-label">Assigned Carrier</div>
            <div class="sidebar-tenant-name">{user_airline} Airlines</div>
            <div class="sidebar-tenant-user">Operator: {current_user['username']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tenant Scope Switcher for Super Admin
    if is_super_admin:
        scope_options = ["All Airlines"] + available_airlines
        active_airline_scope = st.selectbox("Airline Viewport Scope", scope_options, index=0)
    else:
        active_airline_scope = user_airline
        st.caption(f"Tenant isolation active: Scoped to {user_airline}.")

    st.markdown("---")

    # Only 3 essential modules (AI Engine removed as requested)
    nav_selection = st.radio(
        "Navigation",
        [
            "Executive KPI Dashboard",
            "AI Review Triage & Live Router",
            "Support Ticket Operations Center"
        ]
    )

    st.markdown("---")

    # Backend Connectivity Indicator
    if API_ONLINE:
        st.caption("Service Connection: REST API Active")
    else:
        st.caption("Service Connection: Local Direct Mode")

    if st.button("Sign Out", use_container_width=True, type="secondary"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()


# Filter Dataset according to Tenant Scope
if active_airline_scope == "All Airlines":
    tenant_df = raw_df
else:
    tenant_df = raw_df[raw_df[airline_col].astype(str).str.lower() == active_airline_scope.lower()] if not raw_df.empty else pd.DataFrame()


# ==============================================================================
# 1. EXECUTIVE KPI DASHBOARD (BLUE HEADING & PSYCHOLOGICAL HIERARCHY)
# ==============================================================================
if nav_selection == "Executive KPI Dashboard":
    # Requested specific blue title
    if active_airline_scope == "All Airlines":
        header_text = "Global Airline Feedback & Operations Overview"
    else:
        header_text = f"{active_airline_scope} - Executive Feedback & Operations Overview"

    st.markdown(f'<div class="brand-title">{header_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Real-time operational sentiment, risk signals, and complaint velocity monitoring.</div>', unsafe_allow_html=True)

    ticket_stats = get_ticket_statistics(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)

    if tenant_df.empty:
        st.warning(f"No historical review records found for '{active_airline_scope}'. Live reviews will populate as they arrive.")
    else:
        total_rev = len(tenant_df)
        neg_pct = (tenant_df["airline_sentiment"] == "negative").mean() * 100
        pos_pct = (tenant_df["airline_sentiment"] == "positive").mean() * 100
        neu_pct = (tenant_df["airline_sentiment"] == "neutral").mean() * 100
        net_sentiment_score = pos_pct - neg_pct  # NSS Metric (-100 to +100)

        # Psychological Alert: If Dissatisfaction Rate is high, alert operations team immediately
        if neg_pct > 50.0:
            st.markdown(f"""
            <div class="executive-alert">
                <strong>Operational Advisory:</strong> Negative feedback is currently at <strong>{neg_pct:.1f}%</strong> of total volume for <strong>{active_airline_scope}</strong>. Primary drivers require automated triage to prevent passenger churn and regulatory SLA penalties.
            </div>
            """, unsafe_allow_html=True)

        # High-Impact KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Reviews Analyzed", f"{total_rev:,}", f"Scope: {active_airline_scope}")
        k2.metric("Dissatisfaction Rate", f"{neg_pct:.1f}%", f"{int(total_rev * neg_pct / 100):,} complaints", delta_color="inverse")
        k3.metric("Customer Praise Rate", f"{pos_pct:.1f}%", f"{int(total_rev * pos_pct / 100):,} compliments")
        k4.metric("Net Sentiment Score", f"{net_sentiment_score:+.1f}", "Target: > +10.0")
        k5.metric("Open Action Tickets", f"{ticket_stats.get('open_tickets', 0)}", f"{ticket_stats.get('total_tickets', 0)} Total Created", delta_color="off")

        st.markdown("---")

        # Visual Analytics Row
        col_left, col_right = st.columns([1, 1.25])

        with col_left:
            st.markdown("##### Sentiment Distribution Breakdown")
            sent_counts = tenant_df["airline_sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]

            fig_sentiment = px.pie(
                sent_counts,
                values="Count",
                names="Sentiment",
                hole=0.5,
                color="Sentiment",
                color_discrete_map={
                    "negative": "#DC2626",  # Red
                    "neutral": "#64748B",   # Slate neutral
                    "positive": "#16A34A"   # Green
                }
            )
            fig_sentiment.update_traces(textposition="inside", textinfo="percent+label")
            fig_sentiment.update_layout(margin=dict(t=15, b=15, l=15, r=15), height=310)
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col_right:
            if active_airline_scope == "All Airlines":
                st.markdown("##### Cross-Carrier Passenger Sentiment Comparison")
                airline_sent = raw_df.groupby([airline_col, "airline_sentiment"]).size().reset_index(name="Count")
                fig_airline = px.bar(
                    airline_sent,
                    x=airline_col,
                    y="Count",
                    color="airline_sentiment",
                    barmode="stack",
                    color_discrete_map={
                        "negative": "#DC2626",
                        "neutral": "#64748B",
                        "positive": "#16A34A"
                    },
                    labels={airline_col: "Airline", "Count": "Review Count"}
                )
                fig_airline.update_layout(margin=dict(t=15, b=15, l=15, r=15), height=310, legend_title_text="Sentiment")
                st.plotly_chart(fig_airline, use_container_width=True)
            else:
                st.markdown(f"##### Primary Complaint Categories ({active_airline_scope})")
                reason_col = "negativereason_clean" if "negativereason_clean" in tenant_df.columns else "negativereason"
                neg_df = tenant_df[tenant_df["airline_sentiment"] == "negative"]

                if reason_col in neg_df.columns and not neg_df[reason_col].dropna().empty:
                    top_reasons = neg_df[reason_col].value_counts().head(8).reset_index()
                    top_reasons.columns = ["Complaint Reason", "Count"]
                    fig_reasons = px.bar(
                        top_reasons,
                        x="Count",
                        y="Complaint Reason",
                        orientation="h",
                        color="Count",
                        color_continuous_scale="Blues",
                    )
                    fig_reasons.update_layout(yaxis=dict(autorange="reversed"), height=310, margin=dict(t=15, b=15, l=15, r=15))
                    st.plotly_chart(fig_reasons, use_container_width=True)
                else:
                    st.info("No categorized negative complaint records logged for this carrier.")

        # Interactive Review Explorer
        with st.expander(f"Carrier Feedback Audit Ledger ({active_airline_scope})", expanded=False):
            search_query = st.text_input("Filter records by operational keyword (e.g. luggage, delayed, refund, gate, staff):", key="search_kw")
            display_df = tenant_df
            text_col = "clean_text" if "clean_text" in display_df.columns else "text"
            if search_query:
                display_df = display_df[display_df[text_col].astype(str).str.contains(search_query.lower(), na=False)]

            cols_to_show = [c for c in ["tweet_id", airline_col, text_col, "airline_sentiment", "negativereason"] if c in display_df.columns]
            st.dataframe(display_df[cols_to_show].head(150), use_container_width=True, height=260)


# ==============================================================================
# 2. AI REVIEW TRIAGE & LIVE ROUTER (PSYCHOLOGICAL INCIDENT TRIAGE DESK)
# ==============================================================================
elif nav_selection == "AI Review Triage & Live Router":
    st.markdown('<div class="brand-title">AI Review Triage & Incident Router</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Real-time discourse clause decomposition, sub-issue sentiment classification, and automated department dispatch.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("##### Incoming Passenger Feedback")

        review_input = st.text_area(
            "Passenger Feedback Statement:",
            value="",
            placeholder="Type or paste passenger review here (e.g. My flight was delayed by 3 hours and luggage was lost in Chicago)...",
            height=140
        )

        # Airline Selection
        if is_super_admin:
            airline_selected = st.selectbox("Target Carrier:", available_airlines + ["Other"], index=0)
        else:
            airline_selected = user_airline
            st.info(f"Target carrier locked to {user_airline} Airlines.")

        auto_ticket = st.checkbox("Automatically dispatch actionable issues as support tickets to operational queue", value=True)
        analyze_btn = st.button("Run AI Triage & Department Dispatch", type="primary", use_container_width=True)

    with col2:
        st.markdown("##### Immediate Triage Classification")

        if review_input.strip():
            analysis = api_predict_and_route(
                text=review_input.strip(),
                airline=airline_selected,
                create_tickets=(auto_ticket and analyze_btn)
            )

            sent = analysis.get("overall_sentiment", analysis.get("sentiment", "neutral"))
            conf = analysis.get("overall_confidence", analysis.get("confidence", 0.85))
            probs = analysis.get("overall_probabilities", analysis.get("probabilities", {"negative": 0.33, "neutral": 0.33, "positive": 0.34}))

            sent_color = "#DC2626" if sent == "negative" else "#64748B" if sent == "neutral" else "#16A34A"

            st.markdown(
                f"""
                <div class="triage-box" style="border-left-color: {sent_color};">
                    <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Overall Review Sentiment</div>
                    <div style="color: {sent_color}; font-size: 1.45rem; font-weight: 800; margin: 3px 0 6px 0;">{sent.upper()}</div>
                    <div style="font-size: 0.84rem; color: #334155;">Model Confidence: <b>{conf*100:.1f}%</b> | Identified Clauses: <b>{analysis.get('total_issues_detected', 1)}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            prob_df = pd.DataFrame([
                {"Sentiment": "Negative", "Probability": probs.get("negative", 0.0)},
                {"Sentiment": "Neutral", "Probability": probs.get("neutral", 0.0)},
                {"Sentiment": "Positive", "Probability": probs.get("positive", 0.0)}
            ])
            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Sentiment",
                orientation="h",
                color="Sentiment",
                color_discrete_map={"Negative": "#DC2626", "Neutral": "#64748B", "Positive": "#16A34A"},
                text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_prob.update_layout(xaxis_range=[0, 1], height=150, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 2.2rem 1.5rem; text-align: center; color: #64748B;">
                <div style="font-weight: 600; font-size: 0.95rem; color: #334155; margin-bottom: 4px;">Awaiting Passenger Feedback</div>
                <div style="font-size: 0.85rem;">Enter passenger feedback text on the left to trigger real-time AI sentiment analysis, discourse clause decomposition, and department dispatch.</div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Sub-Clause Breakdown
    if review_input.strip():
        st.markdown("---")
        st.markdown("##### Sub-Clause Decomposition & Department Dispatch Matrix")

        issues = analysis.get("extracted_issues", [])
        if not issues:
            st.info("No actionable operational issues isolated in this statement.")
        else:
            cols = st.columns(len(issues) if len(issues) <= 4 else 3)
            for idx, issue in enumerate(issues):
                c = cols[idx % len(cols)]
                p_class = f"badge-{issue['priority'].lower()}"

                s_color = "#DC2626" if issue['sentiment'] == "negative" else "#64748B" if issue['sentiment'] == "neutral" else "#16A34A"
                s_bg = "#FEE2E2" if issue['sentiment'] == "negative" else "#F1F5F9" if issue['sentiment'] == "neutral" else "#DCFCE7"

                dept_conf = issue.get('department_confidence', 0.85) * 100
                sent_conf = issue.get('sentiment_confidence', 0.85) * 100

                with c:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-weight: 700; color: #1D4ED8; font-size: 0.95rem;">{issue['department']}</span>
                                <span class="{p_class}">{issue['priority']}</span>
                            </div>
                            <div style="font-size: 0.88rem; color: #1E293B; font-weight: 500; margin-bottom: 8px; line-height: 1.35;">"{issue['issue_description']}"</div>
                            <div style="font-size: 0.82rem; color: #475569; background: #F8FAFC; padding: 6px 8px; border-radius: 4px; border: 1px solid #E2E8F0; margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                    <span>Clause Polarity:</span>
                                    <span style="color: {s_color}; font-weight: 700; background: {s_bg}; padding: 1px 6px; border-radius: 3px;">{issue['sentiment'].title()} ({sent_conf:.1f}%)</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Routing Precision:</span>
                                    <span style="color: #1D4ED8; font-weight: 700;">{dept_conf:.1f}%</span>
                                </div>
                            </div>
                            <div style="font-size: 0.76rem; color: #64748B;">
                                <b>Action Required:</b> {'Ticket Dispatched' if issue['is_actionable'] else 'Compliment Logged'}<br/>
                                <b>Engine:</b> {issue.get('routing_engine', 'Meta Zero-Shot NLI')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if auto_ticket and analyze_btn and analysis.get("generated_tickets"):
                st.success(f"Dispatched {len(analysis['generated_tickets'])} support ticket(s) to {airline_selected} operational queue.")


# ==============================================================================
# 3. SUPPORT TICKET OPERATIONS CENTER (BLUE HEADING AS REQUESTED)
# ==============================================================================
elif nav_selection == "Support Ticket Operations Center":
    # Requested specific blue title
    if active_airline_scope == "All Airlines":
        ticket_header_text = "Support Ticket Operations Center (All Airline Tenants)"
    else:
        ticket_header_text = f"Support Ticket Operations Center ({active_airline_scope})"

    st.markdown(f'<div class="brand-title">{ticket_header_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Operational queue for passenger complaint tracking, department assignment, and SLA resolution notes.</div>', unsafe_allow_html=True)

    ticket_stats = get_ticket_statistics(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)

    # High-Visibility SLA Status Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Incidents", ticket_stats.get("total_tickets", 0))
    k2.metric("Open / Unassigned", ticket_stats.get("status_breakdown", {}).get("Open", 0), "Needs Immediate Action")
    k3.metric("Under Investigation", ticket_stats.get("status_breakdown", {}).get("In Progress", 0), "In Progress")
    k4.metric("Resolved / Closed", ticket_stats.get("status_breakdown", {}).get("Resolved", 0) + ticket_stats.get("status_breakdown", {}).get("Closed", 0), "Resolved SLA")
    k5.metric("Urgent Escalations", ticket_stats.get("priority_breakdown", {}).get("URGENT", 0), "Critical SLA", delta_color="inverse")

    st.markdown("---")

    # Filters Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        dept_filter = st.selectbox(
            "Department Filter",
            ["All", "Flight Operations", "Baggage Services", "Reservations & Ticketing", "Customer Experience", "In-flight Services", "Digital Support"]
        )
    with f2:
        status_filter = st.selectbox("Status Filter", ["All", "Open", "In Progress", "Resolved", "Closed"])
    with f3:
        priority_filter = st.selectbox("Priority Filter", ["All", "URGENT", "HIGH", "MEDIUM", "LOW"])
    with f4:
        if is_super_admin and active_airline_scope == "All Airlines":
            airline_filter = st.selectbox("Airline Filter", ["All"] + available_airlines)
        else:
            airline_filter = active_airline_scope
            st.text_input("Tenant Scope", value=active_airline_scope, disabled=True)

    tickets = api_list_tickets(
        department=dept_filter,
        status=status_filter,
        priority=priority_filter,
        airline=airline_filter,
        limit=250
    )

    if not tickets:
        st.info(f"No incident tickets found matching the selected filters for '{active_airline_scope}'.")
    else:
        df_tickets = pd.DataFrame(tickets)

        st.dataframe(
            df_tickets[["ticket_id", "airline", "priority", "department", "status", "issue_description", "created_at", "resolution_notes"]],
            use_container_width=True,
            height=280
        )

        st.markdown("---")
        st.markdown("##### Incident Action Desk")

        u_col1, u_col2, u_col3 = st.columns([1, 1, 2])
        with u_col1:
            ticket_to_update = st.selectbox("Select Ticket Identifier:", df_tickets["ticket_id"].tolist())
        with u_col2:
            new_status = st.selectbox("Update Status To:", ["Open", "In Progress", "Resolved", "Closed"])
        with u_col3:
            res_notes = st.text_input("Operational Resolution Notes:", placeholder="e.g. Passenger contacted, luggage located at carousel 4, voucher issued.")

        b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 2])
        with b_col1:
            if st.button("Commit Status Update", type="primary", use_container_width=True):
                updated = api_update_ticket_status(ticket_to_update, new_status, res_notes)
                if updated:
                    st.success(f"Ticket {ticket_to_update} successfully transitioned to '{new_status}'.")
                    st.rerun()

        with b_col2:
            if st.button("Delete Ticket Record", type="secondary", use_container_width=True):
                if api_delete_ticket(ticket_to_update):
                    st.warning(f"Ticket {ticket_to_update} deleted from database.")
                    st.rerun()

        with b_col3:
            clear_label = f"Purge All {active_airline_scope} Tickets" if active_airline_scope != "All Airlines" else "Purge All Tickets (Admin)"
            if st.button(clear_label, use_container_width=True):
                api_delete_all_tickets(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)
                st.info("Target tickets cleared.")
                st.rerun()
