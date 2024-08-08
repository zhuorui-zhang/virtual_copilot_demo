import streamlit as st
from src.st_pages import add_page_title, get_nav_from_toml

# st.set_page_config(layout="wide")
st.set_page_config()

# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`
nav = get_nav_from_toml(".streamlit/pages.toml")

st.logo("src/st_pages/logo_home.png")

pg = st.navigation(nav)

add_page_title(pg)

pg.run()