from manim import *

# define colors
c1 = rgb_to_color([0.0, 0.4470, 0.7410])  # blue
c2 = rgb_to_color([0.8500, 0.3250, 0.0980])  # red
c3 = rgb_to_color([0.9290, 0.6940, 0.1250])  # yellow
c4 = rgb_to_color([0.4940, 0.1840, 0.5560])  # purple
c5 = rgb_to_color([0.4660, 0.6740, 0.1880])  # green
c6 = rgb_to_color([0.3010, 0.7450, 0.9330])  # cyan
c7 = rgb_to_color([0.6350, 0.0780, 0.1840])  # dark red
c8 = rgb_to_color([99 / 255, 200 / 255, 166 / 255])
c9 = rgb_to_color([72 / 255, 107 / 255, 135 / 255])
c10 = rgb_to_color([248 / 255, 148 / 255, 125 / 255])


# color tinting function
def color_tint(base_color, percent):
    return interpolate_color(WHITE, base_color, percent)


# Vertical text class
class VerticalText(VGroup):
    def __init__(self, text, font_size=36, color=WHITE, buff=0.1, **kwargs):
        super().__init__(**kwargs)
        for char in text:
            if char == " ":
                # 用一个不可见的小空白占位
                letter = Text(" ", font_size=font_size).scale(0.1)
            else:
                letter = Text(char, font_size=font_size, color=color)
            self.add(letter)
        self.arrange(DOWN, buff=buff)


class Logo(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        title_en1 = Text("BEAUTY", font_size=26, color=color_tint(c1, 0.6))
        title_en2 = Text("of", font_size=26, color=color_tint(c2, 0.6))
        title_en3 = Text("MATRIX", font_size=26, color=color_tint(c3, 0.6))
        title_en2.next_to(title_en1, DOWN, buff=0.1)
        title_en3.next_to(title_en2, DOWN, buff=0.1)
        title_en_group = VGroup(title_en1, title_en2, title_en3)
        title_en_group.to_edge(DR, buff=0.5)
        self.add(title_en_group)
