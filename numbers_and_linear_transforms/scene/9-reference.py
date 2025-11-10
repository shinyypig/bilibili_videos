from manim import *
from utils import *
import numpy as np


class ReferenceScene(ThreeDScene):
    def construct(self):
        logo = Logo()
        self.add(logo)

        title = Text("参考文献", font_size=38, color=WHITE)
        title.to_edge(UL, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)

        text = Text(
            "耿修瑞. 矩阵之美（基础篇）[M]. 北京: 科学出版社, 2023. ISBN 978-7-03-074944-4.",
            font_size=22,
            color=WHITE,
        )
        text.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(Write(text))

        text2 = Text(
            "思考题：什么样的实矩阵（方阵）可以在实数域开任意次方？",
            font_size=28,
            color=WHITE,
        )
        text2.next_to(text, DOWN, buff=1.2).to_edge(LEFT, buff=0.5)
        self.play(Write(text2))
