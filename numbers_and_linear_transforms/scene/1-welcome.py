from manim import *
from utils import *


class WelcomeScene(Scene):
    def construct(self):

        title = Text("数理同契", font_size=80, color=WHITE)
        title.move_to(ORIGIN)

        logo_cn = VerticalText("矩阵之美", font_size=32, color=WHITE)
        logo = Logo()
        title_group = VGroup(logo_cn, logo)

        logo.next_to(logo_cn, RIGHT, aligned_edge=DOWN)
        title_group.to_edge(DR, buff=0.5)

        self.play(
            Write(title),
            run_time=2,
            subcaption="各位同学早！这里是矩阵之美",
            subcaption_duration=4,
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
            subcaption="今天我们来聊一聊线性变换",
            subcaption_duration=4,
        )
        self.wait(2)
