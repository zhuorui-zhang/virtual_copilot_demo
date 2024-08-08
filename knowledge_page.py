import streamlit as st
import utils
from knowledge_graph.AIcode3 import toNeo4j_v2

# 获取当前屏幕的宽度和高度
screen_height, screen_width = utils.get_screen_HW()
print(screen_width, screen_height)
# 计算宽度和高度的一半
iframe_width = int(screen_width) // 2
iframe_height = int(screen_height) // 2

text_intro = "*\"Knowledge graphs enhance pilot assistance by integrating flight schedules,\
                weather, air traffic, and aircraft data. This enables real-time insights, \
                predictive analytics, and comprehensive situational awareness, improving safety, \
                efficiency, and decision-making for pilots.\"*"
text_intro = utils.set_text(text_intro, font_size=18, font_color="blue", font_weight='normal')

st.markdown(text_intro, unsafe_allow_html=True)

st.image("knowledge_graph/graph_concept.png", use_column_width=True)
st.markdown('#')

with st.container(border=True):
    text_try = ':rainbow[👇 click  to have a try]'
    text_try = utils.set_text(text_try, font_size=18, font_weight='normal', text_align='center')
    st.markdown(text_try, unsafe_allow_html=True)
    # 嵌入具有自己端口号的子页面
    # subpage_url = "http://localhost:8888"  # 替换为实际的子页面URL和端口号
    # 添加一个按钮，点击后跳转到指定URL
    build_button = st.button("💡 Build Knowledge Graph", use_container_width=True)
    utils.ChangeButtonColour('💡 Build Knowledge Graph', 'black', '#ACF4A2')
    if build_button:
        utils.ChangeButtonColour('💡 Build Knowledge Graph', 'black', '#FC5252')
        with st.spinner('Building Knowledge Graph...'):
            toNeo4j_v2.main()
            st.success('Graph Build Successful!', icon="✅")
            st.balloons()
        subpage_url = 'http://localhost:7474/browser/'  # 替换为实际的URL
        # st.write(f"[Finished! Click Here To Watch Your Results~]({subpage_url})")
        st.link_button("Finished! Click Here To View Your Results~", subpage_url, type='primary')
        # Neo4j donot support to be embedded in iframe, you need other tricks to enabel x-frame-options
        # st.components.v1.iframe(subpage_url, width=iframe_width, height=iframe_height)


    # st.image("knowledge_graph/balloons_crop.png", use_column_width=True)

