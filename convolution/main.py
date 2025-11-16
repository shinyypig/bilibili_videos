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
        ## Show continuous convolution definition
        title = Text("连续函数的卷积：", font_size=26, color=WHITE)

        formula = MathTex(
            r"(f*g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau",
            color=WHITE,
            font_size=30,
        ).next_to(title, RIGHT, buff=1)

        vgroup1 = VGroup(title, formula).to_edge(UP, buff=0.5).set_x(0)
        self.play(Write(vgroup1), run_time=4)

        def func_f(t):
            return np.exp(-0.5 * t**2)

        def func_g(t):
            return (t <= 1) * (t >= 0) * t

        def func_fg(t, tau):
            return func_f(t) * func_g(tau - t)

        def func_fg_integral(tau):
            tlist = np.linspace(-3.5, 3.5, 1000)
            dt = tlist[1] - tlist[0]
            integral = np.sum(func_f(tlist) * func_g(tau - tlist)) * dt
            return integral

        axes_f = Axes(x_range=[-5, 5, 1], y_range=[0, 1.5], x_length=7, y_length=2)
        f_graph = axes_f.plot(
            func_f, color=c1, x_range=[-3.5, 3.5, 0.1], use_smoothing=False
        )
        axes_labels = axes_f.get_axis_labels(x_label="t")
        graphs = VGroup(axes_f, axes_labels, f_graph)
        graphs.next_to(vgroup1, DOWN, buff=0.1)
        self.play(FadeIn(graphs), run_time=2)

        g_graph = axes_f.plot(
            func_g, color=c2, x_range=[-3.5, 3.5, 0.1], use_smoothing=False
        )
        graphs.add(g_graph)
        self.play(FadeIn(g_graph), run_time=2)
        self.wait(2)
        self.play(FadeOut(g_graph), run_time=1)

        tau = ValueTracker(-2)
        g_tau = always_redraw(
            lambda: axes_f.plot(
                lambda t: func_g(tau.get_value() - t),
                color=c2,
                x_range=[
                    -3.5,
                    3.5,
                    0.01,
                ],
                use_smoothing=False,
            )
        )
        self.play(FadeIn(g_tau), run_time=2)
        self.wait(1)

        axes_c = Axes(x_range=[-5, 5, 1], y_range=[0, 1.5], x_length=7, y_length=2)
        axes_c_labels = axes_c.get_axis_labels(x_label=r"t")

        fg_tau = always_redraw(
            lambda: axes_c.plot(
                lambda t: func_fg(t, tau.get_value()),
                color=c3,
                x_range=[-3.5, tau.get_value(), 0.01],
                use_smoothing=False,
            )
        )
        fg_area = always_redraw(
            lambda: axes_c.get_area(
                fg_tau,
                x_range=[-3.5, 3.5],
                color=c3,
                opacity=0.7,
            )
        )

        conv_graph = always_redraw(
            lambda: axes_c.plot(
                lambda t: func_fg_integral(t),
                color=c4,
                x_range=[-3.5, tau.get_value(), 0.01],
                use_smoothing=False,
            )
        )

        conv_graph_group = VGroup(axes_c, axes_c_labels, fg_tau, fg_area, conv_graph)
        conv_graph_group.next_to(graphs, DOWN, buff=0.5)

        self.play(FadeIn(conv_graph_group), run_time=2)

        self.play(
            tau.animate.set_value(4),
            run_time=8,
            rate_func=linear,
        )
        self.wait(2)


class DiscreteConvolution(Scene):
    def build_stem_group(self, axes, samples, color):
        """Create a group of stems (lollipops) for discrete samples."""
        group = VGroup()
        for position, value in samples:
            start = axes.c2p(position, 0)
            end = axes.c2p(position, value)
            stem = Line(start, end, color=color, stroke_width=5)
            dot = Dot(end, color=color, radius=0.05)
            group.add(VGroup(stem, dot))
        return group

    def discrete_conv(self, x_samples, h_samples):
        """Return convolution result as ordered list of (n, value)."""
        x_keys = sorted(x_samples.keys())
        h_keys = sorted(h_samples.keys())
        n_min = x_keys[0] + h_keys[0]
        n_max = x_keys[-1] + h_keys[-1]
        result = []
        for n in range(n_min, n_max + 1):
            total = 0
            for k, xv in x_samples.items():
                h_idx = n - k
                if h_idx in h_samples:
                    total += xv * h_samples[h_idx]
            result.append((n, total))
        return result

    def construct(self):
        ## Show discrete convolution definition
        title = Text("离散序列的卷积：", font_size=26, color=WHITE)
        formula = MathTex(
            r"(x*h)[n] = \sum_{k=-\infty}^{\infty} x[k]\;h[n-k]",
            font_size=30,
            color=WHITE,
        ).next_to(title, RIGHT, buff=1)
        vgroup1 = VGroup(title, formula).to_edge(UP, buff=0.5).set_x(0)
        self.play(Write(vgroup1), run_time=4)

        # Sample sequences
        x_samples = {-1: 1, 0: 2, 1: 3, 2: 2, 3: 1}
        h_samples = {0: 0.7, 1: 1, 2: 0.5}
        h_flipped = {-k: v for k, v in h_samples.items()}

        axes_config = dict(
            x_range=[-3, 6, 1],
            y_range=[0, 5, 1],
            x_length=8,
            y_length=2,
            axis_config={"include_tip": False},
        )

        axes_x = Axes(**axes_config)
        axes_h = Axes(**axes_config)
        axes_x.next_to(vgroup1, DOWN, buff=0.7)
        axes_h.next_to(axes_x, DOWN, buff=1)

        x_axis_label = MathTex("x[n]", color=c1).next_to(axes_x, LEFT, buff=0.3)
        h_axis_label = MathTex("h[n]", color=c2).next_to(axes_x, RIGHT, buff=0.3)
        conv_axis_label = MathTex("(x*h)[n]", color=c4).next_to(axes_h, LEFT, buff=0.3)
        labels = VGroup(x_axis_label, h_axis_label, conv_axis_label)

        self.play(
            LaggedStart(
                FadeIn(axes_x, shift=UP),
                FadeIn(axes_h, shift=UP),
                lag_ratio=0.2,
            ),
            run_time=1,
        )
        self.play(FadeIn(labels), run_time=1)

        x_group = self.build_stem_group(axes_x, sorted(x_samples.items()), color=c1)
        h_group = self.build_stem_group(axes_x, sorted(h_samples.items()), color=c2)
        self.play(
            LaggedStart(*[FadeIn(stem) for stem in x_group], lag_ratio=0.1),
            run_time=1,
        )
        self.play(
            LaggedStart(*[FadeIn(stem) for stem in h_group], lag_ratio=0.1),
            run_time=1,
        )

        # Flip h[n] -> h[-k] to prepare for sliding
        flipped_label = MathTex("h[-k]", color=c2).next_to(axes_x, RIGHT, buff=0.3)
        h_flipped_group = self.build_stem_group(
            axes_x, sorted(h_flipped.items()), color=c2
        )
        self.play(Transform(h_group, h_flipped_group), run_time=1)
        self.play(Transform(h_axis_label, flipped_label), run_time=1)

        # Setup sliding animation with tracker
        n_tracker = ValueTracker(-1)

        def get_shifted_samples():
            shift = n_tracker.get_value()
            samples = [(k + shift, v) for k, v in sorted(h_flipped.items())]
            return samples

        shifted_h = always_redraw(
            lambda: self.build_stem_group(axes_x, get_shifted_samples(), color=c2)
        )
        shifted_label = MathTex("h[n-k]", color=c2).next_to(axes_x, RIGHT, buff=0.3)
        self.play(FadeOut(h_group))
        self.play(FadeIn(shifted_h), Transform(h_axis_label, shifted_label), run_time=1)

        # Display convolution build-up
        # n_text = MathTex("n = -1", color=c4).to_corner(UR)
        # value_text = MathTex(
        #     r"(x*h)[n] = \sum x[k]h[n-k]",
        #     color=c4,
        #     font_size=32,
        # ).next_to(axes_h, RIGHT, buff=0.5)
        # self.play(Write(n_text), Write(value_text))

        conv_values = self.discrete_conv(x_samples, h_samples)

        for idx, (n_val, conv_val) in enumerate(conv_values):
            # new_n_text = MathTex(f"n = {n_val}", color=c4).to_corner(UR)
            # self.play(Transform(n_text, new_n_text), run_time=0.6)
            self.play(
                n_tracker.animate.set_value(n_val),
                run_time=0.5,
                rate_func=smooth,
            )

            # calc_text = MathTex(
            #     rf"(x*h)[{n_val}] = {conv_val}",
            #     color=c4,
            #     font_size=32,
            # ).next_to(axes_h, RIGHT, buff=0.5)
            # self.play(Transform(value_text, calc_text), run_time=0.6)

            stem = self.build_stem_group(axes_h, [(n_val - 1, conv_val)], color=c4)
            self.play(FadeIn(stem, shift=UP), run_time=0.6)

        self.wait(2)
