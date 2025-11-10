from manim import *
from utils import *
import numpy as np


class CompleteScene(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo)

        title = Text("代数完备", font_size=38, color=WHITE)
        title.to_edge(UL, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)

        text1 = Text("是否存在其他线性变换？", font_size=30, color=WHITE)
        text1.to_edge(UP, buff=1)
        text2 = Text("不存在！", font_size=30, color=WHITE)
        text2.next_to(text1, DOWN, buff=0.2)
        self.play(Write(text1))
        self.wait(0.4)
        self.play(Write(text2))

        box = SurroundingRectangle(VGroup(text1, text2), buff=0.1, color=c1)
        self.play(Create(box))

        text3 = Text("证明", font_size=30, color=WHITE)
        text3.next_to(box, DOWN, buff=0.5)
        self.play(Write(text3))

        text4 = Text("问题等价于是否存在非复数特征值", font_size=30, color=WHITE)
        text4.next_to(text2, DOWN, buff=1.3)
        self.play(Write(text4))

        tex = MathTex(
            r"\det(\mathbf{T} - \lambda \mathbf{I}) = 0", font_size=36, color=WHITE
        )
        tex.next_to(text4, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(tex))

        text5 = Text("复数域代数完备", font_size=30, color=WHITE)
        text5.next_to(tex, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(text5))

        tex2 = MathTex(r"\lambda \in \mathbb{C}", font_size=36, color=WHITE)
        tex2.next_to(text5, RIGHT, buff=0.3)
        self.play(Write(tex2))

        text6 = Text("证毕", font_size=30, color=WHITE)
        text6.next_to(text5, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(text6))

        box2 = SurroundingRectangle(
            VGroup(text3, text4, tex, text5, tex2, text6), buff=0.2, color=c2
        )
        self.play(Create(box2))
        self.wait(1)
