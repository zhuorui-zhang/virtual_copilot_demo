import os
import streamlit as st
import utils
from knowledge_graph.AIcode3 import toNeo4j_v2 as toNeo4j_v2_old
from knowledge_graph.demonstration import toNeo4j_v2, wordclass
import pandas as pd


with open("knowledge_graph/demonstration/90-FMS Engine Failure Procedures/FMS Engine Failure Procedures .txt", "r") as f:
    fm_txt = f.read()
    fm_txt = fm_txt.replace('\n', '').replace('\t', '')

# 获取当前屏幕的宽度和高度
screen_height, screen_width = utils.get_screen_HW()
print(screen_width, screen_height)
# 计算宽度和高度的一半
iframe_width = int(screen_width) // 2
iframe_height = int(screen_height) // 2

col1, col2, col3 = st.columns([0.25,0.5,0.25])
with col2:
    st.title('📖 Expert Knowledge Graph for Pilot Assistance')
    st.caption("🚀 powered by Neo4j")
    text_intro = "*\"Knowledge graphs enhance pilot assistance by integrating flight schedules,\
                    weather, air traffic, and aircraft data. This enables real-time insights, \
                    predictive analytics, and comprehensive situational awareness, improving safety, \
                    efficiency, and decision-making for pilots.\"*"
    text_intro = utils.set_text(text_intro, font_size=18, font_color="blue", font_weight='normal')

    st.markdown(text_intro, unsafe_allow_html=True)

    st.image("knowledge_graph/graph_concept.png", use_column_width=True)
    st.markdown('#')
    
    

    with st.container(border=True):
        
        fm_txt = utils.set_text(fm_txt, font_size=18, font_weight="normal", font_color='#b45427')
        st.markdown(fm_txt, unsafe_allow_html=True)
        text_try = ':rainbow[👇 click  to have a try]'
        text_try = utils.set_text(text_try, font_size=18, font_weight='normal', text_align='center')
        st.markdown(text_try, unsafe_allow_html=True)
        col1, col2 = st.columns([0.5, 0.5])
        # 嵌入具有自己端口号的子页面
        # subpage_url = "http://localhost:8888"  # 替换为实际的子页面URL和端口号
        # 添加一个按钮，点击后跳转到指定URL
        with col1:
            build_button_1 = st.button("💡 Build Text To My Knowledge Graph", use_container_width=True)
        utils.ChangeButtonColour('💡 Build Text To My Knowledge Graph', 'black', '#ACF4A2')
        if build_button_1:
            utils.ChangeButtonColour('💡 Build Text To My Knowledge Graph', 'black', '#FC5252')
            with st.spinner('Build Text To My Knowledge Graph...'):
                 # 第一步：词性统计
                print('*'*10+'第一步：词性统计'+'*'*10)
                wordclass_output_path='wordclass_output_path.xlsx'  #词汇统计数据导出路径
                file_path = r"90-FMS Engine Failure Procedures/FMS Engine Failure Procedures .txt"                
                fig = wordclass.fun_wordclass(file_path, wordclass_output_path, show_plot=False)
                
                #第二步:三元组显示
                print('*'*10+'第二步:三元组显示'+'*'*10)
                sanyuanzu_path="knowledge_graph/demonstration/三元组.xlsx"
                df_sanyuanzu=pd.read_excel(sanyuanzu_path)
                print(df_sanyuanzu)

                #第三步：相似词合并
                print('*'*10+'第三步：相似词合并'+'*'*10)
                xiangsici_path = "knowledge_graph/demonstration/90-FMS Engine Failure Procedures/新知识合并结果.xlsx"

                excel_file = pd.ExcelFile(xiangsici_path)
                # 获取所有工作表的名称
                sheet_names = excel_file.sheet_names
                # 打印工作表名称及其内容
                for sheet in sheet_names:
                    print(f'Sheet name: {sheet}')
                    # 读取当前工作表的内容
                    df = excel_file.parse(sheet)
                    # 打印当前工作表的内容
                    print(df)
                    print('\n')
                #第四步：生成知识图谱
                print('*'*10+'第四步：生成知识图谱'+'*'*10)

                toNeo4j_v2.main(xiangsici_path)
                st.success('Graph Build Successful!', icon="✅")
                st.balloons()
                st.pyplot(fig)

            subpage_url = 'http://localhost:7474/browser/'  # 替换为实际的URL
            # st.write(f"[Finished! Click Here To Watch Your Results~]({subpage_url})")
            st.link_button("Finished! Click Here To View Your Results~", subpage_url, type='primary')
            # Neo4j donot support to be embedded in iframe, you need other tricks to enabel x-frame-options
            # st.components.v1.iframe(subpage_url, width=iframe_width, height=iframe_height)

        with col2:
            # 添加一个按钮，点击后跳转到指定URL
            build_button_2 = st.button("💡 Build An Interactive Graph", use_container_width=True)
        utils.ChangeButtonColour('💡 Build An Interactive Graph', 'black', '#f2e651')
        if build_button_2:
            utils.ChangeButtonColour('💡 Build An Interactive Graph', 'black', '#FC5252')
            with st.spinner('Building An Interactive Graph...'):
                toNeo4j_v2_old.main()
                st.success('Graph Build Successful!', icon="✅")
                st.balloons()
                st.pyplot(fig)

            subpage_url = 'http://localhost:7474/browser/'  # 替换为实际的URL
            # st.write(f"[Finished! Click Here To Watch Your Results~]({subpage_url})")
            st.link_button("Finished! Click Here To View Your Results~", subpage_url, type='primary')
            # Neo4j donot support to be embedded in iframe, you need other tricks to enabel x-frame-options
            # st.components.v1.iframe(subpage_url, width=iframe_width, height=iframe_height)
