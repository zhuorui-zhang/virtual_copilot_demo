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