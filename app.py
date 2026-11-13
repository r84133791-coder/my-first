import streamlit as st

st.title("ሰላም! ከ VS Code የሠራሁት ስትሪምሊት አፕ 🚀")
st.write("ይህ በጣም አሪፍ እና ቀላል ነው።")

name = st.text_input("ስምዎን ያስገቡ:")
if name:
    st.success(f"እንኳን ደህና መጡ፣ {name}!")