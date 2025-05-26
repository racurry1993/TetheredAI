import streamlit as st

st.title("My Basic Website")
st.header("Welcome to my site")
st.write("This is a simple website created using Streamlit on an Android tablet")
if st.button("Click Me"):
    st.write("You clicked the button.")
