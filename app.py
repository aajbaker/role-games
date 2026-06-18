import streamlit as st

st.set_page_config(page_title="Role Games", layout="wide")

pg = st.navigation([
    st.Page("Play.py", title="Play"),
    st.Page("Simulate.py", title="Simulate"),
])

# Clear any in-progress game when navigating to Play from another page
if pg.title == "Play" and st.session_state.get("_active_page") != "Play":
    st.session_state.pop("g", None)
st.session_state["_active_page"] = pg.title

pg.run()
