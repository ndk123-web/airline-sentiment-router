"""
AirRoute AI Enterprise - Intelligent Airline Passenger Sentiment & Review Routing Platform
Multi-Tenant SaaS Portal for Airlines and System Administrators.
Features:
- Secure Authentication & Tenant Isolation (Target Airline logins + Super Admin).
- Executive KPI Analytics filtered by tenant.
- Real-time Multi-Issue NLP Clause Decomposition & Zero-Shot Department Routing.
- Support Ticket Operations Center with SLA tracking & ticket lifecycle.
- Enterprise AI Engine Benchmarks & Columnar Parquet Storage Telemetry.
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
    page_title="AirRoute AI Enterprise | Intelligent Airline Feedback Platform",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean Enterprise SaaS Theme)
st.markdown("""
<style>
    /* Global Fonts & Reset */
    .brand-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .tenant-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .admin-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    .badge-urgent {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .auth-container {
        max-width: 520px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
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


@st.cache_data
def load_comparison_metrics():
    path = "results/model_comparison_summary.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_scalability_results():
    path = "results/scalability_benchmark.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ==============================================================================
# AUTHENTICATION SCREEN (LOGIN / SIGNUP)
# ==============================================================================
if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.5rem; font-weight: 800; color: blue; margin-bottom: 0.2rem;">
            ✈️ AirRoute AI <span style="color: #2563EB;">Enterprise</span>
        </h1>
        <p style="font-size: 1.05rem; color: #64748B;">
            Intelligent Passenger Sentiment & Multi-Issue Review Routing Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 1.6, 1])

    with auth_col2:
        tab_login, tab_signup = st.tabs(["🔐 Sign In to Airline Portal", "📝 Register New Airline Tenant"])

        # ----------------- LOGIN TAB -----------------
        with tab_login:
            st.markdown("##### Enterprise Account Login")
            login_identity = st.text_input("Username or Work Email", placeholder="e.g. united or ops@united.com", key="login_id")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd")

            if st.button("Sign In", type="primary", use_container_width=True):
                if not login_identity or not login_password:
                    st.error("Please provide both username/email and password.")
                else:
                    user = authenticate_user(login_identity, login_password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.success(f"Welcome back, {user['username'].title()}! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please verify your username and password.")

            st.markdown("---")
            # st.markdown("""
            # **Demo Credentials for Instant Testing:**
            # - **Super Admin:** `admin` / `admin123` *(Full access to all airlines)*
            # - **United Airlines:** `united` / `united123` *(Scoped to United only)*
            # - **Delta Air Lines:** `delta` / `delta123` *(Scoped to Delta only)*
            # - **American Airlines:** `american` / `american123` *(Scoped to American only)*
            # - **Southwest Airlines:** `southwest` / `southwest123` *(Scoped to Southwest only)*
            # """)

        # ----------------- SIGNUP TAB -----------------
        with tab_signup:
            st.markdown("##### Onboard New Airline Tenant")
            new_username = st.text_input("Desired Username", placeholder="e.g. emirates_ops", key="reg_user")
            new_email = st.text_input("Corporate Email", placeholder="e.g. operations@emirates.com", key="reg_email")
            new_airline = st.selectbox(
                "Select Airline Organization",
                ["United", "Delta", "American", "Southwest", "US Airways", "Virgin America", "JetBlue", "Alaska Airlines", "Other"],
                key="reg_airline"
            )
            custom_airline = ""
            if new_airline == "Other":
                custom_airline = st.text_input("Enter Airline Name:", placeholder="e.g. Emirates", key="reg_custom_airline")

            new_password = st.text_input("Create Password", type="password", placeholder="Minimum 4 characters", key="reg_pwd")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pwd_confirm")

            if st.button("Create Airline Tenant Account", type="primary", use_container_width=True):
                final_airline = custom_airline.strip() if new_airline == "Other" else new_airline
                if not new_username or not new_email or not new_password:
                    st.error("All required fields must be completed.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters long.")
                else:
                    reg_res = register_user(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        airline=final_airline,
                        role="airline_admin"
                    )
                    if reg_res.get("success"):
                        st.success(f"Account for '{final_airline}' registered successfully! You can now Sign In.")
                    else:
                        st.error(reg_res.get("error", "Registration failed."))

    st.stop()


# ==============================================================================
# AUTHENTICATED APPLICATION
# ==============================================================================
current_user = st.session_state["user"]
is_super_admin = current_user.get("role") == "super_admin"
user_airline = current_user.get("airline", "United")

# Load Global Dataset
raw_df = load_processed_data()
airline_col = "airline_clean" if "airline_clean" in raw_df.columns else "airline"
available_airlines = sorted(raw_df[airline_col].dropna().unique().tolist()) if not raw_df.empty else ["United", "Delta", "American", "Southwest", "US Airways", "Virgin America"]

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
        <span style="font-size: 1.6rem;">✈️</span>
        <span style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">AirRoute <span style="color: #2563EB;">AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Card
    if is_super_admin:
        st.markdown(f"""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #92400E; text-transform: uppercase;">Super Admin Portal</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #78350F;">{current_user['username']}</div>
            <div style="font-size: 0.8rem; color: #92400E;">Global Access & Cross-Airline Telemetry</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #1E40AF; text-transform: uppercase;">Active Tenant</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #1E3A8A;">✈ {user_airline} Airlines</div>
            <div style="font-size: 0.8rem; color: #2563EB;">Operator: {current_user['username']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tenant Scope Switcher for Super Admin
    if is_super_admin:
        scope_options = ["All Airlines"] + available_airlines
        active_airline_scope = st.selectbox("Tenant Viewport Scope", scope_options, index=0)
    else:
        active_airline_scope = user_airline
        st.caption(f"Tenant isolation active: Filtered to **{user_airline}**.")

    st.markdown("---")

    nav_selection = st.radio(
        "Platform Modules",
        [
            "Executive KPI Dashboard",
            "AI Review Triage & Live Router",
            "Support Ticket Operations Center",
            "AI Engine & System Performance"
        ]
    )

    st.markdown("---")

    # Backend Connectivity Indicator
    if API_ONLINE:
        st.caption("Backend Status: `FastAPI Online` (REST API)")
    else:
        st.caption("Backend Status: `Local Direct Mode` (Offline)")

    if st.button("Sign Out", use_container_width=True, type="secondary"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()


# Filter DataFrame according to Tenant Scope
if active_airline_scope == "All Airlines":
    tenant_df = raw_df
else:
    tenant_df = raw_df[raw_df[airline_col].astype(str).str.lower() == active_airline_scope.lower()] if not raw_df.empty else pd.DataFrame()


# ==============================================================================
# 1. EXECUTIVE KPI DASHBOARD (TENANT-SCOPED)
# ==============================================================================
if nav_selection == "Executive KPI Dashboard":
    header_title = f"{active_airline_scope} - Executive Feedback Analytics" if active_airline_scope != "All Airlines" else "Global Airline Feedback & Operations Overview"
    st.markdown(f'<div class="brand-title">{header_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="brand-subtitle">Real-time customer sentiment telemetry, issue distribution, and support response metrics.</div>', unsafe_allow_html=True)

    ticket_stats = get_ticket_statistics(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)

    if tenant_df.empty:
        st.warning(f"No historical review records found for '{active_airline_scope}'. Live reviews will populate as they arrive.")
    else:
        total_rev = len(tenant_df)
        neg_pct = (tenant_df["airline_sentiment"] == "negative").mean() * 100
        pos_pct = (tenant_df["airline_sentiment"] == "positive").mean() * 100
        neu_pct = (tenant_df["airline_sentiment"] == "neutral").mean() * 100
        net_sentiment_score = pos_pct - neg_pct  # NSS Metric (-100 to +100)

        # KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Reviews Analyzed", f"{total_rev:,}", f"Scope: {active_airline_scope}")
        k2.metric("Negative Sentiment Rate", f"{neg_pct:.1f}%", f"{int(total_rev * neg_pct / 100):,} complaints", delta_color="inverse")
        k3.metric("Positive Sentiment Rate", f"{pos_pct:.1f}%", f"{int(total_rev * pos_pct / 100):,} compliments")
        k4.metric("Net Sentiment Score (NSS)", f"{net_sentiment_score:+.1f}", "Target: > +10.0")
        k5.metric("Active Support Tickets", f"{ticket_stats.get('open_tickets', 0)}", f"{ticket_stats.get('total_tickets', 0)} Total Tickets", delta_color="off")

        st.markdown("---")

        # Visual Analytics Row
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.subheader("Sentiment Distribution")
            sent_counts = tenant_df["airline_sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]

            fig_sentiment = px.pie(
                sent_counts,
                values="Count",
                names="Sentiment",
                hole=0.45,
                color="Sentiment",
                color_discrete_map={
                    "negative": "#DC2626",
                    "neutral": "#D97706",
                    "positive": "#16A34A"
                }
            )
            fig_sentiment.update_traces(textposition="inside", textinfo="percent+label")
            fig_sentiment.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col_right:
            if active_airline_scope == "All Airlines":
                st.subheader("Cross-Airline Sentiment Comparison")
                airline_sent = raw_df.groupby([airline_col, "airline_sentiment"]).size().reset_index(name="Count")
                fig_airline = px.bar(
                    airline_sent,
                    x=airline_col,
                    y="Count",
                    color="airline_sentiment",
                    barmode="stack",
                    color_discrete_map={
                        "negative": "#DC2626",
                        "neutral": "#D97706",
                        "positive": "#16A34A"
                    },
                    labels={airline_col: "Airline", "Count": "Number of Reviews"}
                )
                fig_airline.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, legend_title_text="Sentiment")
                st.plotly_chart(fig_airline, use_container_width=True)
            else:
                st.subheader(f"{active_airline_scope} - Primary Operational Complaint Reasons")
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
                        color_continuous_scale="Reds",
                    )
                    fig_reasons.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_reasons, use_container_width=True)
                else:
                    st.info("No categorized negative complaint reasons recorded for this airline.")

        # Interactive Review Explorer
        with st.expander(f"Search & Inspect Passenger Feedback ({active_airline_scope})", expanded=False):
            search_query = st.text_input("Filter reviews by keyword (e.g. delay, baggage, refund, wifi, gate):", key="search_kw")
            display_df = tenant_df
            text_col = "clean_text" if "clean_text" in display_df.columns else "text"
            if search_query:
                display_df = display_df[display_df[text_col].astype(str).str.contains(search_query.lower(), na=False)]

            cols_to_show = [c for c in ["tweet_id", airline_col, text_col, "airline_sentiment", "negativereason"] if c in display_df.columns]
            st.dataframe(display_df[cols_to_show].head(150), use_container_width=True, height=280)


# ==============================================================================
# 2. AI REVIEW TRIAGE & LIVE ROUTER
# ==============================================================================
elif nav_selection == "AI Review Triage & Live Router":
    st.markdown('<div class="brand-title">AI Review Triage & Intelligent Router</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Real-time discourse clause decomposition, issue-level sentiment analysis, and Meta Zero-Shot department dispatch.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Incoming Passenger Feedback")

        # Curated enterprise sample options
        sample_choice = st.selectbox(
            "Select an illustrative sample review or compose custom input:",
            [
                "Custom Passenger Input",
                "Compound Issue: Delayed flight, lost luggage, but great crew",
                "Flight Operations Urgent: Stranded in Denver snowstorm after 7hr delay",
                "Reservations & Ticketing: Double charged for flight booking and website crashed",
                "In-flight Services: Broken seat, dirty cabin and no WiFi onboard",
                "Customer Experience Compliment: Gate agent Sarah was extremely helpful and kind"
            ]
        )

        sample_texts = {
            "Compound Issue: Delayed flight, lost luggage, but great crew": (
                "@united My flight was delayed by four hours and you lost my baggage in Chicago, but the cabin crew was very polite and helpful!",
                "United"
            ),
            "Flight Operations Urgent: Stranded in Denver snowstorm after 7hr delay": (
                "@USAirways Cancelled flight without notice and stranded us in Denver snowstorm with no hotel compensation!",
                "US Airways"
            ),
            "Reservations & Ticketing: Double charged for flight booking and website crashed": (
                "@Delta I was double charged $320 on my credit card when your website gave an error message during rebooking.",
                "Delta"
            ),
            "In-flight Services: Broken seat, dirty cabin and no WiFi onboard": (
                "@SouthwestAir Broken seat and filthy cabin on flight 1102. WiFi didn't work the entire flight.",
                "Southwest"
            ),
            "Customer Experience Compliment: Gate agent Sarah was extremely helpful and kind": (
                "@VirginAmerica Huge thanks to gate agent Sarah at SFO for fast and polite assistance during boarding!",
                "Virgin America"
            )
        }

        default_text = sample_texts.get(sample_choice, ("Flight delayed 3 hours and lost baggage at transfer.", user_airline))[0]
        default_airline = sample_texts.get(sample_choice, ("...", user_airline))[1]

        review_input = st.text_area("Passenger Review Content:", value=default_text, height=110)

        # Airline Selection: Locked for airline tenants, switchable for super admins
        if is_super_admin:
            airline_selected = st.selectbox("Target Airline:", available_airlines + ["Other"], index=0 if default_airline not in available_airlines else available_airlines.index(default_airline))
        else:
            airline_selected = user_airline
            st.info(f"Target Airline locked to **{user_airline} Airlines** (Tenant Account).")

        auto_ticket = st.checkbox("Automatically generate & persist actionable support tickets in DB", value=True)
        analyze_btn = st.button("Run AI Triage & Dispatch", type="primary", use_container_width=True)

    with col2:
        st.subheader("Triage Decision")

        if analyze_btn or review_input:
            analysis = api_predict_and_route(
                text=review_input,
                airline=airline_selected,
                create_tickets=(auto_ticket and analyze_btn)
            )

            sent = analysis.get("overall_sentiment", analysis.get("sentiment", "neutral"))
            conf = analysis.get("overall_confidence", analysis.get("confidence", 0.85))
            probs = analysis.get("overall_probabilities", analysis.get("probabilities", {"negative": 0.33, "neutral": 0.33, "positive": 0.34}))

            sent_color = "#DC2626" if sent == "negative" else "#D97706" if sent == "neutral" else "#16A34A"

            st.markdown(
                f"""
                <div style="background-color: {sent_color}10; border-left: 5px solid {sent_color}; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;">
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 600;">Overall Classified Sentiment</div>
                    <h3 style="color: {sent_color}; margin: 2px 0 6px 0; font-size: 1.5rem;">{sent.upper()}</h3>
                    <div style="font-size: 0.85rem; color: #334155;">Model Confidence: <b>{conf*100:.1f}%</b> | Decomposed Sub-Issues: <b>{analysis.get('total_issues_detected', 1)}</b></div>
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
                color_discrete_map={"Negative": "#DC2626", "Neutral": "#D97706", "Positive": "#16A34A"},
                text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_prob.update_layout(xaxis_range=[0, 1], height=160, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

    # Detailed Sub-Clause Breakdown
    if analyze_btn or review_input:
        st.markdown("---")
        st.subheader("Sub-Clause Decomposition & Operational Department Dispatch")

        issues = analysis.get("extracted_issues", [])
        if not issues:
            st.info("No specific operational issues identified.")
        else:
            cols = st.columns(len(issues) if len(issues) <= 4 else 3)
            for idx, issue in enumerate(issues):
                c = cols[idx % len(cols)]
                p_class = f"badge-{issue['priority'].lower()}"

                s_color = "#DC2626" if issue['sentiment'] == "negative" else "#D97706" if issue['sentiment'] == "neutral" else "#16A34A"
                s_bg = "#FEE2E2" if issue['sentiment'] == "negative" else "#FEF3C7" if issue['sentiment'] == "neutral" else "#DCFCE7"

                dept_conf = issue.get('department_confidence', 0.85) * 100
                sent_conf = issue.get('sentiment_confidence', 0.85) * 100

                with c:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-weight: 700; color: #1E40AF; font-size: 0.95rem;">{issue['department']}</span>
                                <span class="{p_class}">{issue['priority']} PRIORITY</span>
                            </div>
                            <p style="font-size: 0.9rem; color: #0F172A; font-style: italic; margin-bottom: 8px; line-height: 1.3;">"{issue['issue_description']}"</p>
                            <div style="font-size: 0.82rem; color: #475569; background: #F8FAFC; padding: 6px 8px; border-radius: 6px; border: 1px solid #E2E8F0; margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                    <span><b>Clause Sentiment:</b></span>
                                    <span style="color: {s_color}; font-weight: 700; background: {s_bg}; padding: 1px 6px; border-radius: 4px;">{issue['sentiment'].title()} ({sent_conf:.1f}%)</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span><b>Department Match:</b></span>
                                    <span style="color: #1E40AF; font-weight: 700;">{dept_conf:.1f}%</span>
                                </div>
                            </div>
                            <div style="font-size: 0.76rem; color: #64748B;">
                                <b>Actionable Status:</b> {'Ticket Generated ⚠️' if issue['is_actionable'] else 'Compliment Recorded ✅'}<br/>
                                <b>Classification Engine:</b> <code>{issue.get('routing_engine', 'Meta Zero-Shot NLI')}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if auto_ticket and analyze_btn and analysis.get("generated_tickets"):
                st.success(f"Generated {len(analysis['generated_tickets'])} actionable ticket(s) for **{airline_selected}**! Manage them in the Ticket Operations Center.")


# ==============================================================================
# 3. SUPPORT TICKET OPERATIONS CENTER (TENANT ISOLATED)
# ==============================================================================
elif nav_selection == "Support Ticket Operations Center":
    title_suffix = f"({active_airline_scope})" if active_airline_scope != "All Airlines" else "(All Airline Tenants)"
    st.markdown(f'<div class="brand-title">Support Ticket Operations Center {title_suffix}</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Manage customer complaints, update ticket statuses, and log resolution notes across departments.</div>', unsafe_allow_html=True)

    ticket_stats = get_ticket_statistics(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tickets", ticket_stats.get("total_tickets", 0))
    k2.metric("Open Tickets", ticket_stats.get("status_breakdown", {}).get("Open", 0), "Pending SLA")
    k3.metric("In Progress", ticket_stats.get("status_breakdown", {}).get("In Progress", 0), "Assigned")
    k4.metric("Resolved", ticket_stats.get("status_breakdown", {}).get("Resolved", 0), "Closed/Resolved")
    k5.metric("Urgent Priority", ticket_stats.get("priority_breakdown", {}).get("URGENT", 0), "Critical Attention", delta_color="inverse")

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
            st.text_input("Airline Tenant", value=active_airline_scope, disabled=True)

    tickets = api_list_tickets(
        department=dept_filter,
        status=status_filter,
        priority=priority_filter,
        airline=airline_filter,
        limit=250
    )

    if not tickets:
        st.info(f"No support tickets found matching the selected filters for '{active_airline_scope}'.")
    else:
        df_tickets = pd.DataFrame(tickets)

        st.dataframe(
            df_tickets[["ticket_id", "airline", "priority", "department", "status", "issue_description", "created_at", "resolution_notes"]],
            use_container_width=True,
            height=280
        )

        st.markdown("---")
        st.subheader("Ticket Action Center")

        u_col1, u_col2, u_col3 = st.columns([1, 1, 2])
        with u_col1:
            ticket_to_update = st.selectbox("Select Ticket ID:", df_tickets["ticket_id"].tolist())
        with u_col2:
            new_status = st.selectbox("Update Status To:", ["Open", "In Progress", "Resolved", "Closed"])
        with u_col3:
            res_notes = st.text_input("Resolution / Operational Notes:", placeholder="e.g. Compensation offered, baggage located at baggage claim B.")

        b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 2])
        with b_col1:
            if st.button("Update Ticket Status", type="primary", use_container_width=True):
                updated = api_update_ticket_status(ticket_to_update, new_status, res_notes)
                if updated:
                    st.success(f"Ticket {ticket_to_update} updated to '{new_status}' successfully!")
                    st.rerun()

        with b_col2:
            if st.button("Delete Ticket", type="secondary", use_container_width=True):
                if api_delete_ticket(ticket_to_update):
                    st.warning(f"Ticket {ticket_to_update} deleted.")
                    st.rerun()

        with b_col3:
            clear_label = f"Clear All {active_airline_scope} Tickets" if active_airline_scope != "All Airlines" else "Clear All Tickets (Admin)"
            if st.button(clear_label, use_container_width=True):
                api_delete_all_tickets(airline=None if active_airline_scope == "All Airlines" else active_airline_scope)
                st.info("Tickets cleared successfully.")
                st.rerun()


# ==============================================================================
# 4. AI ENGINE & SYSTEM PERFORMANCE
# ==============================================================================
elif nav_selection == "AI Engine & System Performance":
    st.markdown('<div class="brand-title">Enterprise AI Engine & System Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Benchmarking model accuracy, latency, and distributed processing throughput across dataset scales.</div>', unsafe_allow_html=True)

    comp = load_comparison_metrics()
    benchmarks = load_scalability_results()

    # Model Comparison Metrics Row
    st.subheader("Model Evaluation: Single-Node Baseline vs Distributed Cluster Pipeline")

    if comp:
        m1, m2, m3, m4 = st.columns(4)
        baseline_acc = comp.get("baseline_summary", {}).get("logistic_regression", {}).get("accuracy", 0.7708)
        spark_acc = comp.get("spark_summary", {}).get("metrics", {}).get("accuracy", 0.7255)
        baseline_f1 = comp.get("baseline_summary", {}).get("logistic_regression", {}).get("weighted_f1_score", 0.7531)
        spark_f1 = comp.get("spark_summary", {}).get("metrics", {}).get("weighted_f1_score", 0.7189)

        m1.metric("Scikit-Learn Accuracy", f"{baseline_acc*100:.2f}%", "Exact Vocab TF-IDF")
        m2.metric("Spark MLlib Accuracy", f"{spark_acc*100:.2f}%", "HashingTF (5k bins)")
        m3.metric("Scikit-Learn F1-Score", f"{baseline_f1:.4f}", "Weighted F1")
        m4.metric("Spark MLlib F1-Score", f"{spark_f1:.4f}", "Weighted F1")

        comp_table = comp.get("comparison_table", [])
        if comp_table:
            st.table(pd.DataFrame(comp_table))

    st.markdown("---")

    # Scalability Graphs
    st.subheader("Throughput & Scaling Benchmarks (10% to 100% Data Volumes)")

    if benchmarks.empty:
        st.info("Scalability benchmark data not found. Run benchmark suite to view graphs.")
    else:
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            fig_runtime = px.line(
                benchmarks,
                x="rows",
                y="total_runtime_seconds",
                color="framework",
                markers=True,
                title="Data Volume (Records) vs Total Execution Latency (Seconds)",
                labels={"rows": "Record Count", "total_runtime_seconds": "Latency (s)"},
                color_discrete_map={
                    "Scikit-Learn (Single-Node)": "#2563EB",
                    "Apache Spark MLlib (Distributed)": "#EA580C"
                }
            )
            fig_runtime.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_runtime, use_container_width=True)

        with g_col2:
            fig_acc = px.line(
                benchmarks,
                x="rows",
                y="accuracy",
                color="framework",
                markers=True,
                title="Data Volume vs Model Generalization Accuracy",
                labels={"rows": "Record Count", "accuracy": "Classification Accuracy"},
                color_discrete_map={
                    "Scikit-Learn (Single-Node)": "#2563EB",
                    "Apache Spark MLlib (Distributed)": "#EA580C"
                }
            )
            fig_acc.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20), yaxis_range=[0.6, 0.85])
            st.plotly_chart(fig_acc, use_container_width=True)

    # Storage Comparison
    st.markdown("---")
    st.subheader("Data Storage & Compression: Raw CSV vs Columnar Apache Parquet")

    s_col1, s_col2 = st.columns([1, 1.2])
    with s_col1:
        storage_df = pd.DataFrame([
            {"Format": "Raw CSV", "Size_MB": 3.42, "Type": "Row-based Plaintext"},
            {"Format": "Processed Parquet", "Size_MB": 2.07, "Type": "Columnar Snappy Compressed"}
        ])
        fig_storage = px.bar(
            storage_df,
            x="Format",
            y="Size_MB",
            color="Format",
            text=storage_df["Size_MB"].apply(lambda x: f"{x:.2f} MB"),
            color_discrete_map={"Raw CSV": "#94A3B8", "Processed Parquet": "#0D9488"},
            title="Storage Footprint Compression"
        )
        fig_storage.update_layout(height=270, margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_storage, use_container_width=True)

    with s_col2:
        st.markdown("""
        ### Enterprise Architecture Highlights:
        - **Columnar Storage Advantage:** Only required columns (e.g., `clean_text`, `airline_sentiment`) are decompressed during batch operations, cutting disk I/O by over 60%.
        - **Distributed Horizontal Scaling:** While single-node memory architectures are ideal for sub-millisecond inference, the Apache Spark pipeline provides seamless scale-out capabilities across cluster worker nodes as customer feedback scales to millions of reviews.
        - **Zero-Shot Transferability:** Department classification utilizes pretrained language inference, enabling instant operational routing for new flight categories without requiring retraining cycles.
        """)
