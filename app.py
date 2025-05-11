import streamlit as st
import webbrowser

# Set page configuration
st.set_page_config(page_title="HirePlus AI Services", layout="wide")

# Title of the landing page
st.title("Welcome to HirePlus AI Services Landing Page")

# Define button details
buttons = [
    {
        "label": "Resume Analyzer",
        "image_url": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fdevquestcareers.com%2Fassets%2Fresume-analyzer.jpg&f=1&nofb=1&ipt=866b729e7ae43904010972337a3d2002b819e40af7170cbb4c04996efa9de659",
        "link": "https://hireplusresumeanalyzer.streamlit.app/"
    },
    {
        "label": "Recommender System",
        "image_url": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fpng.pngtree.com%2Fpng-clipart%2F20191121%2Foriginal%2Fpngtree-human-silhouette-avatar-and-stars-job-vector-icon-png-image_5140927.jpg&f=1&nofb=1&ipt=8c2f92e34dbcb2ac1b53a12b99ec8f15f0d45a30d1c0c243dce1531dc7ea5d58",  # Replace with actual image URL
        "link": "https://hireplusrecommendersystem.streamlit.app/"
    },
    {
        "label": "HireBot",
        "image_url": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.istockphoto.com%2Fvectors%2Fchat-bot-ai-and-customer-service-support-concept-vector-flat-person-vector-id1221348467%3Fk%3D20%26m%3D1221348467%26s%3D612x612%26w%3D0%26h%3Dhp8h8MuGL7Ay-mxkmIKUsk3RY4O69MuiWjznS_7cCBw%3D&f=1&nofb=1&ipt=39e2d25ae52611efa2598b2dc973be1f96ba3e7874740f5cef9b0454d39b56e0",
        "link": "https://hirebot.streamlit.app/"
    }
]

# Create a horizontal grid for the buttons
cols = st.columns(len(buttons))

for col, button in zip(cols, buttons):
    with col:
        st.image(button["image_url"], use_container_width=True)
        if st.button(button["label"], use_container_width=True):
            webbrowser.open_new_tab(button["link"])
            st.rerun()