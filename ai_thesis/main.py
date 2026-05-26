# import necessary libraries
from manim import *
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from utils import *


# Scene 1: Welcome Scene
class WelcomeScene(Scene):
    def construct(self):
        ## Show the welcome title and logo
        main_title = Text("Vibe Coding 实战", font_size=80, color=WHITE)
        sub_title = Text("一周完成本科毕业设计", font_size=40, color=WHITE)

        title = VGroup(main_title, sub_title).arrange(
            DOWN, buff=0.5, aligned_edge=RIGHT
        )
        title.move_to(ORIGIN)

        logo_cn = VerticalText("矩阵之美", font_size=32, color=WHITE)
        logo = Logo()
        title_group = VGroup(logo_cn, logo)

        logo.next_to(logo_cn, RIGHT, aligned_edge=DOWN)
        title_group.to_edge(DR, buff=0.5)

        self.play(
            Write(title),
            run_time=2,
        )
        self.play(
            FadeIn(title_group, shift=UP),
            run_time=2,
        )
        self.play(
            FadeOut(logo_cn),
            FadeOut(title),
            shift=UP,
            run_time=2,
        )
        self.wait(2)
