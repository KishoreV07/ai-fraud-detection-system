"""
auth.py
--------
Simple password-based authentication for the Streamlit dashboard.
Not enterprise-grade security — suitable for a demo/portfolio project
to satisfy an "Admin Login" requirement without adding a full user
database and hashing infrastructure.

For a real production system, use proper password hashing (bcrypt),
a users table in the database, and session tokens instead.
"""

import streamlit as st

# In a real system, load this from an environment variable or secrets.toml,
# never hardcode credentials. For Streamlit Cloud: Settings > Secrets.
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "fraud2026"  # CHANGE THIS before sharing your app publicly


def check_login() -> bool:
    """
    Shows a login form if the user isn't authenticated yet.
    Returns True if logged in, False otherwise (and renders the form).
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.subheader("🔒 Admin Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

    return False


def logout_button():
    """Renders a logout button in the sidebar if the user is logged in."""
    if st.session_state.get("authenticated"):
        if st.sidebar.button("Log out"):
            st.session_state.authenticated = False
            st.rerun()