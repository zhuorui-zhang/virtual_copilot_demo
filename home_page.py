import streamlit as st
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

    
st.title('📈 展望未来: AI-based COMMUNICATION in AVIATION')
# st.markdown("#")

st.markdown('### 🚩 我们的愿景:')
col1, col2 = st.columns([0.6,0.4])
with col1:
    text = "💡 To develop a virtual co-pilot system that can assist the pilot in various tasks, \
        such as monitoring the aircraft, communicating with air traffic control, and handling emergencies. \
        This system will enhance safety, efficiency, and decision-making for pilots, and enable the transition to single-pilot aircraft."
    text = "💡 助力未来低空飞行，提高飞行过程的安全性+效率，\
        例如增强态势感知、与空中交通管制的高质量通信以及紧急情况决策辅助。\
        该系统将利用AI语言模型，结合VOLTE通信技术，提高飞行员的决策能力，最终实现向人机共驾的过渡。"
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
col1, col2, col3 = st.columns([0.3,0.3,0.4], vertical_alignment="center", gap='large')

with col1:
    st.image("./src/imgs/poster.png")
with col2:
    # st.image("./src/imgs/data_samples.png")
    st.image("./src/imgs/two_pilots_resize.png")

st.markdown("###")
st.markdown('### 🤔 如何保障未来低空飞行通讯质量?')
text = "💡 背景分析: \
根据航空领域的统计数据，绝大部分空中事故源自人为失误，而其中一个核心因素便是空管-机组间通信系统的效率与可靠性不足。\
在低空飞行场景中，飞行器受到复杂地形、天气、设备干扰等多重因素的影响，通信效率的提升对于保障飞行安全至关重要。"

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
    col1, col2 = st.columns([0.5,0.5], gap="large")
    with col1:
        st.markdown("### 🌞 常规飞行场景的通信需求")

        # text1 = "⭐ Ground inspection: Whether the windows are closed/Oxygen and fire testing/Communication with the maintenance team."
        # text2 = "⭐ Taxi: Small aircraft conflict/position indication/turn exit instructions/access to ATC/night navigation o auxiliary vision in adverse weather conditions."
        # text3 = "⭐ Climb: Departure program reminder/climb limit or speed limit/boost and other ascent parameter monitoring."
        # text4 = "⭐ Cruise: Changes in communication frequency/inspection of heading and altitude/real-time evaluation of route."
        # text5 = "⭐ Approach: Non Auto landing plan prompt/low altitude ATC prompt/automatic execution of landing checklist."
        text1 = "⭐地面检查：设备状态确认/氧气及防火测试/与维修团队沟通。"
        text2 = "⭐滑行：避免冲突/位置指示/夜间导航、恶劣天气条件下辅助视觉引导。"
        text3 = "⭐爬升：离场程序提醒/爬升限制或速度限制/助推等上升参数监控。"
        text4 = "⭐巡航：通信频率变化/航向和高度检查/航线实时评估。"
        text5 = "⭐进近：非自动着陆计划提示/低空ATC提示/自动执行着陆检查单。"
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
        st.markdown("### 💥 在紧急、异常状况下, 可靠通信需求更加迫切")
        # text = """
        #        - Sudden accidents: Automatically execute electronic checklists (just clarify what the pilot needs to do) To remind or even execute key node information, pilots are only responsible for communicating and conveying instructions to ATC.
        #        - Go around and other procedures: Clarify the required configuration and assist in program execution 
        #        - Bad weather: Cloud Map Intelligent Computing and Path Updating Make a detour decision"
        #        """
        # st.markdown(text)
        # text1 = "⭐ Sudden accidents: Automatically execute electronic checklists (just clarify what the pilot needs to do) To remind or even execute key node information, pilots are only responsible for communicating and conveying instructions to ATC."
        # text2 = "⭐ Go around and other procedures: Clarify the required configuration and assist in program execution."
        # text3 = "⭐ Bad weather: Cloud Map Intelligent Computing and Path Updating Make a detour decision"
        text1 = "⭐突发事故：自动执行电子检查单（只需明确飞行员需要做什么）来提醒甚至执行关键节点信息，飞行员只负责向ATC沟通和传达指令。"
        text2 = "⭐复飞等程序：明确所需复飞程序与机体配置（如襟翼、推力设置），根据实时飞行状态和空域信息重新规划最优航线。"
        text3 = "⭐恶劣天气：结合云图、风切变雷达与人工智能算法，自动推荐绕行航线。"
        text4 = "⭐通信中断：利用备用通信方式（如卫星通信或机载间通信网络）保持联系，AI可在通信中断期间基于规则生成自动化响应。"

        text1 = utils.set_text(text1, font_size=22, font_weight='normal', font_color='black')
        text2 = utils.set_text(text2, font_size=22, font_weight='normal', font_color='black')
        text3 = utils.set_text(text3, font_size=22, font_weight='normal', font_color='black')
        text4 = utils.set_text(text3, font_size=22, font_weight='normal', font_color='black')
        st.markdown(text1, unsafe_allow_html=True)
        st.markdown(text2, unsafe_allow_html=True)
        st.markdown(text3, unsafe_allow_html=True)
        st.markdown(text4, unsafe_allow_html=True)

text = "飞行员最需要怎样的协助？为了更好地助力低空飞行，我们正在做什么？"
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
st.markdown("###")
st.markdown(custom_css_2, unsafe_allow_html=True)
st.markdown(f'<div class="my-container-2">{text}</div>', unsafe_allow_html=True)


st.markdown('### 🛠️ Design and Development')
with st.container(border=True):
    col1, col2 = st.columns([0.5,0.5])
    with col1:
        st.image("src/imgs/vcop1.png")
    with col2:
        st.image("src/imgs/vcop2.png")

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

with st.container(border=True):
    col1, col2, col3 = st.columns([0.2,0.6,0.2])
    with col2:
        st.image("src/imgs/combined_all.png")

