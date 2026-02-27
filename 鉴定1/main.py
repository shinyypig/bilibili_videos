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
        main_title = Text("热门视频鉴定", font_size=80, color=WHITE)
        subtitle = Text("雷达天线做Wi-Fi？", font_size=40, color=WHITE)
        subtitle.next_to(main_title, DOWN, buff=0.5, aligned_edge=RIGHT)
        title = VGroup(main_title, subtitle)
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
