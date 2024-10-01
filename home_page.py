import streamlit as st
import cv2
import utils

# import streamlit_extras
# from streamlit_extras.stylable_container import stylable_container 

# # Define your custom CSS #f0f2f6
# custom_css = """
# <style>
# .my-container {
# background-color:  #f0f2f6;
# padding: 10px;
# border-radius: 5px;
# }
# </style>
# """
# text = "To develop a virtual co-pilot system that can assist the pilot in various tasks, \
#         such as monitoring the aircraft, communicating with air traffic control, and handling emergencies. \
#         This system will enhance safety, efficiency, and decision-making for pilots, and enable the transition to single-pilot aircraft."

# st.markdown(custom_css, unsafe_allow_html=True)
# st.markdown(f'<div class="my-container">{text}</div>', unsafe_allow_html=True)

    
st.title('📈 Future Trend: Single-Pilot Aircraft')
# st.markdown("#")

st.markdown('### 🚩 Our vision:')
col1, col2 = st.columns([0.7,0.3])
with col1:
    text = "💡 To develop a virtual co-pilot system that can assist the pilot in various tasks, \
        such as monitoring the aircraft, communicating with air traffic control, and handling emergencies. \
        This system will enhance safety, efficiency, and decision-making for pilots, and enable the transition to single-pilot aircraft."
    text = utils.set_text(text, font_size=24, font_weight='normal', font_color='#7f8386')
    bg_color = "#f0f2f6"
    border_color = "#a9a7a4"  # 设置边界线颜色
    custom_css = f"""
        <style>
        .my-container {{
        background-color: {bg_color};
        padding: 10px;
        border-radius: 10px;
        border: 2px solid {border_color};  # 设置边界线颜色和宽度
        }}
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(f'<div class="my-container">{text}</div>', unsafe_allow_html=True)

st.markdown("###")
col1, col2, col3 = st.columns([0.3,0.5,0.2], vertical_alignment="center", gap='large')

with col1:
    st.image("./src/imgs/poster.png")
with col2:
    st.image("./src/imgs/concepts.png")

st.markdown("###")
st.markdown('### 🤔 What a virtual co-pilot should do?')
text = "💡 The Civil Aviation Administration of China believes that the most common types of human errors\
        in the cockpit can be divided into three categories: incorrect execution, forgetting to execute, and omission of procedures."

text = utils.set_text(text, font_size=24, font_weight='normal', font_color='#7f8386')
bg_color = "#f0f2f6"
border_color = "#a9a7a4"  # 设置边界线颜色
custom_css = f"""
    <style>
    .my-container {{
    background-color: {bg_color};
    padding: 10px;
    border-radius: 10px;
    border: 2px solid {border_color};  # 设置边界线颜色和宽度
    }}
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(f'<div class="my-container">{text}</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns([0.4,0.4,0.2], gap="large")
    with col1:
        st.markdown("### 🌞 Under normal flight conditions")
        # text = """
        #        - Ground inspection: Whether the windows are closed/Oxygen and fire testing/Communication with the maintenance team.
        #        - Taxi: Small aircraft conflict/position indication/turn exit instructions/access to ATC/night navigation o auxiliary vision in adverse weather conditions.
        #        - Climb: Departure program reminder/climb limit or speed limit/boost and other ascent parameter monitoring.
        #        - Cruise: Changes in communication frequency/inspection of heading and altitude/real-time evaluation of route.
        #        - Approach: Non Auto landing plan prompt/low altitude ATC prompt/automatic execution of landing checklist.
        #         """
        # st.markdown(text)

        text1 = "⭐ Ground inspection: Whether the windows are closed/Oxygen and fire testing/Communication with the maintenance team."
        text2 = "⭐ Taxi: Small aircraft conflict/position indication/turn exit instructions/access to ATC/night navigation o auxiliary vision in adverse weather conditions."
        text3 = "⭐ Climb: Departure program reminder/climb limit or speed limit/boost and other ascent parameter monitoring."
        text4 = "⭐ Cruise: Changes in communication frequency/inspection of heading and altitude/real-time evaluation of route."
        text5 = "⭐ Approach: Non Auto landing plan prompt/low altitude ATC prompt/automatic execution of landing checklist."
        text1 = utils.set_text(text1, font_size=22, font_weight='normal', font_color='black')
        text2 = utils.set_text(text2, font_size=22, font_weight='normal', font_color='black')
        text3 = utils.set_text(text3, font_size=22, font_weight='normal', font_color='black')
        text4 = utils.set_text(text4, font_size=22, font_weight='normal', font_color='black')
        text5 = utils.set_text(text5, font_size=22, font_weight='normal', font_color='black')
        st.markdown(text1, unsafe_allow_html=True)
        st.markdown(text2, unsafe_allow_html=True)
        st.markdown(text3, unsafe_allow_html=True)
        st.markdown(text4, unsafe_allow_html=True)
        st.markdown(text5, unsafe_allow_html=True)

    with col2:
        st.markdown("### 💥 Under abnormal circumstances or temporary changes-divided by emergency")
        # text = """
        #        - Sudden accidents: Automatically execute electronic checklists (just clarify what the pilot needs to do) To remind or even execute key node information, pilots are only responsible for communicating and conveying instructions to ATC.
        #        - Go around and other procedures: Clarify the required configuration and assist in program execution 
        #        - Bad weather: Cloud Map Intelligent Computing and Path Updating Make a detour decision"
        #        """
        # st.markdown(text)
        text1 = "⭐ Sudden accidents: Automatically execute electronic checklists (just clarify what the pilot needs to do) To remind or even execute key node information, pilots are only responsible for communicating and conveying instructions to ATC."
        text2 = "⭐ Go around and other procedures: Clarify the required configuration and assist in program execution."
        text3 = "⭐ Bad weather: Cloud Map Intelligent Computing and Path Updating Make a detour decision"

        text1 = utils.set_text(text1, font_size=22, font_weight='normal', font_color='black')
        text2 = utils.set_text(text2, font_size=22, font_weight='normal', font_color='black')
        text3 = utils.set_text(text3, font_size=22, font_weight='normal', font_color='black')
        st.markdown(text1, unsafe_allow_html=True)
        st.markdown(text2, unsafe_allow_html=True)
        st.markdown(text3, unsafe_allow_html=True)

text = "What pilots need to assist with: program checks and prompts/major decision support"
text = utils.set_text(text, font_size=24, font_weight='normal', font_color='green')
bg_color = "#fff4bd"
border_color = "#a9a7a4"  # 设置边界线颜色
custom_css_2 = f"""
    <style>
    .my-container-2 {{
    background-color: {bg_color};
    padding: 10px;
    border-radius: 10px;
    border: 2px solid {border_color};  # 设置边界线颜色和宽度
    }}
    </style>
    """
st.markdown(custom_css_2, unsafe_allow_html=True)
st.markdown(f'<div class="my-container-2">{text}</div>', unsafe_allow_html=True)

st.markdown("###")
st.markdown('### 🛠️ Design and Development')
with st.container(border=True):
    col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], vertical_alignment="center")

with col1:
    st.image("src/imgs/arch_1.jpeg")
with col2:
    st.image("src/imgs/arch_2.png")
with col3:
    st.image("src/imgs/normal_pilot.jpeg")
    text = "Current state: Two-pilots (captain and co-copilot)"
    text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
    caption = st.markdown(text, unsafe_allow_html=True)
with col4:
    st.image("src/imgs/page1_arrow.png")
with col5:
    st.image("src/imgs/copilot.jpeg")
    text = "Future trend: Single-pilot (with a virtual copilot)"
    text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
    caption = st.markdown(text, unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns([0.3, 0.3, 0.4], vertical_alignment="bottom")
    with col1:
        st.image("src/imgs/factors.png")
        text = "The influencing factors of sustainable teamwork between pilot and V-CoP."
        text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
        caption = st.markdown(text, unsafe_allow_html=True)
    with col2:
        st.image("src/imgs/needs.png")
        text = "The needs of a good V-CoP and sustainable teamwork."
        text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
        caption = st.markdown(text, unsafe_allow_html=True)
    with col3:
        st.image("src/imgs/data_integration.png")
        text = "Data Integration in V-CoP."
        text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
        caption = st.markdown(text, unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([0.5, 0.5], vertical_alignment="bottom")
    with col1:
        st.image("src/imgs/case.png")
        text = "The teamwork between Pilot and V-CoP."
        text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
        caption = st.markdown(text, unsafe_allow_html=True)
    with col2:
        st.image("src/imgs/kg.png")
        text = "Feedback and Evaluation."
        text = utils.set_text(text, font_size=16, text_align='center', font_color="#5086ef")
        caption = st.markdown(text, unsafe_allow_html=True)





st.markdown("##")
st.markdown("### 🌍 The needs and future trend of virtual co-pilot")

img_list = ["src/imgs/1.jpeg",
            "src/imgs/2.jpeg",
            "src/imgs/3.jpeg",
            "src/imgs/4.jpeg",
            "src/imgs/5.jpeg"]

with st.container(border=True):
    col1, col2, col3, col4, col5= st.columns([0.2,0.2,0.2,0.2,0.2])

with col1:
    st.image(img_list[0])
with col2:
    st.image(img_list[1])
with col3:
    st.image(img_list[2])
with col4:
    st.image(img_list[3])
with col5:
    st.image(img_list[4])
with st.container(border=True):
    col1, col2, col3 = st.columns([0.2,0.6,0.2])
    with col2:
        st.image("src/imgs/agi.png")