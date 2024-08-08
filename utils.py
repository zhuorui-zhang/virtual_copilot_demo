import tkinter as tk

def get_screen_HW():
    # 创建一个隐藏的Tkinter窗口
    root = tk.Tk()
    root.withdraw()
    # 获取屏幕高度和宽度
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # 关闭Tkinter窗口
    root.destroy()
    # 打印屏幕宽度和高度
    # print(f"屏幕宽度: {screen_width}")
    # print(f"屏幕高度: {screen_height}")
    return screen_height, screen_width

# def set_text(text: str, font_size: int = 24, font_color: str = "black", font_weight: str = "bold") -> str:
    # return f'<span style="font-size: {font_size}px; color: {font_color}; font-weight: {font_weight};">{text}</span>'

def set_text(text: str, font_size: int = 24, font_color: str = "black", font_weight: str = "bold", text_align: str = "left") -> str:
    return f'<span style="font-size: {font_size}px; color: {font_color}; font-weight: {font_weight}; text-align: {text_align}; display: block;">{text}</span>'

import streamlit as st
import streamlit.components.v1 as components

def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

# cols = st.columns(4)
# cols[0].button('first button', key='b1')
# cols[1].button('second button', key='b2')
# cols[2].button('third button', key='b3')
# cols[3].button('fourth button', key='b4')

# ChangeButtonColour('second button', 'red', 'blue') # button txt to find, colour to assign
# ChangeButtonColour('fourth button', '#c19af5', '#354b75') # button txt to find, colour to assign
