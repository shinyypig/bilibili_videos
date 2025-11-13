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
        title = Text("卷积的矩阵表示", font_size=80, color=WHITE)
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


class ContinuousConvolution(Scene):
    def construct(self):
        ##  Define the functions f(t) and g(t)
        f = lambda t: np.exp(-t) * (t >= 0)
        g = lambda t: np.heaviside(t, 0.5) - np.heaviside(t - 1, 0.5)

        # Create axes
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"include_tip": False},
        )

        # Plot f(t)
        f_graph = axes.plot(f, color=BLUE, x_range=[-1, 5])
        f_label = axes.get_graph_label(f_graph, label="f(t)", x_val=4, direction=UP)

        # Plot g(t)
        g_graph = axes.plot(g, color=GREEN, x_range=[-1, 5])
        g_label = axes.get_graph_label(g_graph, label="g(t)", x_val=4, direction=UP)

        # Add graphs to the scene
        self.play(Create(axes))
        self.play(Create(f_graph), Write(f_label))
        self.play(Create(g_graph), Write(g_label))

        # Show convolution integral setup
        convolution_text = MathTex(
            r"(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau"
        ).to_edge(UP)
        self.play(Write(convolution_text))

        # Pause to allow viewers to see the setup
        self.wait(2)
