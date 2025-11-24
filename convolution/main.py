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


class MatrixFormConvolution(Scene):
    def construct(self):
        conv_tex = MathTex(
            r"[1\ 2\ 3\ 4\ 1] * [1\ 2\ 3] = [1 \ 4 \ 10 \ 16 \ 18 \ 14 \ 3]",
            font_size=36,
        )
        conv_tex.to_edge(UP, buff=0.5)
        self.play(Write(conv_tex), run_time=2)

        x_vals = [1, 2, 3, 4, 1]
        h_vals = [1, 2, 3]
        y_vals = np.convolve(x_vals, h_vals)

        # h_text = MathTex(
        #     r"h[n] = [1,\ 2,\ 3]",
        #     color=c2,
        #     font_size=32,
        # ).next_to(conv_tex, DOWN, buff=0.5)
        # x_text = MathTex(
        #     r"x[n] = [1,\ 2,\ 3,\ 4,\ 1]",
        #     color=c1,
        #     font_size=32,
        # ).next_to(h_text, DOWN, buff=0.4)
        # self.play(Write(h_text), Write(x_text), lag_ratio=0.2, run_time=2)

        # build_hint = Text(
        #     "把 h[n] 展成 Toeplitz 矩阵，与 x[n] 列向量相乘",
        #     font_size=28,
        # ).next_to(x_text, DOWN, buff=0.5)
        # self.play(Write(build_hint), run_time=1.4)
        # self.wait(1)

        def build_kernel_matrix(kernel, signal_len):
            rows = len(kernel) + signal_len - 1
            cols = signal_len
            data = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    k_idx = r - c
                    row.append(kernel[k_idx] if 0 <= k_idx < len(kernel) else 0)
                data.append(row)
            return data

        kernel_matrix = build_kernel_matrix(h_vals, len(x_vals))

        # self.play(FadeOut(build_hint), FadeOut(x_text), FadeOut(h_text), run_time=1)

        def entry_to_mob(value):
            color = c2 if value != 0 else color_tint(GRAY_B, 0.4)
            return Integer(value, color=color)

        matrix_mob = Matrix(
            kernel_matrix,
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=entry_to_mob,
        ).scale(0.7)

        x_vector = Matrix(
            [[v] for v in x_vals],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        for entry in x_vector.get_entries():
            entry.set_color(c1)

        result_vector = Matrix(
            [[int(v)] for v in y_vals],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        for entry in result_vector.get_entries():
            entry.set_color(c4)

        times_tex = MathTex(r"\times", font_size=42)
        equals_tex = MathTex(r"=", font_size=42)

        expr_group = VGroup(matrix_mob, times_tex, x_vector, equals_tex, result_vector)
        expr_group.arrange(RIGHT, buff=0.3)
        expr_group.next_to(conv_tex, DOWN, buff=1)

        self.play(Write(matrix_mob), run_time=2)
        self.play(
            FadeIn(times_tex),
            FadeIn(x_vector),
            FadeIn(equals_tex),
            FadeIn(result_vector),
            run_time=1.5,
        )

        # explain = Text(
        #     "矩阵每一行与 x[n] 做内积，正是卷积核的滑动结果",
        #     font_size=28,
        # ).next_to(expr_group, DOWN, buff=0.6)
        # self.play(Write(explain), run_time=1.2)

        matrix_entries = matrix_mob.get_entries()
        rows = len(kernel_matrix)
        cols = len(x_vals)
        row_groups = [
            VGroup(*matrix_entries[i * cols : (i + 1) * cols]) for i in range(rows)
        ]
        result_entries = result_vector.get_entries()

        row_rect = SurroundingRectangle(row_groups[0], buff=0.15, color=c2)
        input_rect = SurroundingRectangle(x_vector, buff=0.2, color=c1)
        result_rect = SurroundingRectangle(result_entries[0], buff=0.2, color=c4)
        self.play(Create(row_rect), Create(input_rect), Create(result_rect), run_time=1)

        n_text = MathTex("n = 0", color=c3, font_size=32).to_corner(UR, buff=0.5)
        # self.play(Write(n_text))

        for idx in range(rows):
            current_row = row_groups[idx]
            current_entry = result_entries[idx]
            n_new = MathTex(f"n = {idx}", color=c3, font_size=32).to_corner(
                UR, buff=0.5
            )
            if idx == 0:
                self.play(
                    # Transform(n_text, n_new),
                    Indicate(current_row, color=c2),
                    Indicate(x_vector, color=c1),
                    Indicate(current_entry, color=c4),
                    run_time=1,
                )
            else:
                self.play(
                    row_rect.animate.move_to(current_row),
                    result_rect.animate.move_to(current_entry),
                    # Transform(n_text, n_new),
                    run_time=0.8,
                )
                self.play(
                    Indicate(current_row, color=c2),
                    Indicate(current_entry, color=c4),
                    run_time=0.6,
                )

        self.wait(2)


class MatrixFormConvolution2(Scene):
    def construct(self):
        conv_tex = MathTex(
            r"\boldsymbol{y} = \boldsymbol{x} * \boldsymbol{h} = \mathbf{H}\boldsymbol{x}",
        )
        conv_tex.to_edge(UP, buff=0.5)
        self.play(Write(conv_tex), run_time=2)

        text = Text(
            "其中，卷积矩阵 H 为 Toeplitz 矩阵",
            font_size=32,
        ).next_to(conv_tex, DOWN, buff=0.5)

        self.play(Write(text), run_time=2)
        self.wait(1)

        H_tex = MathTex(
            r"\mathbf{H} = \begin{bmatrix}"
            r"h_1      & 0        & \cdots & 0        & 0        \\"
            r"h_2      & h_1      & \ddots & \vdots   & \vdots   \\"
            r"h_3      & h_2      & \ddots & 0        & 0        \\"
            r"\vdots   & h_3      & \ddots & h_1      & 0        \\"
            r"h_{m-1}  & \vdots   & \ddots & h_2      & h_1      \\"
            r"h_m      & h_{m-1}  & \ddots & \vdots   & h_2      \\"
            r"0        & h_m      & \ddots & h_{m-2}  & \vdots   \\"
            r"0        & 0        & \ddots & h_{m-1}  & h_{m-2}  \\"
            r"\vdots   & \vdots   & \ddots & h_m      & h_{m-1}  \\"
            r"0        & 0        & 0      & \cdots   & h_m"
            r"\end{bmatrix}",
            font_size=26,
        ).next_to(text, DOWN, buff=0.5)

        self.play(Write(H_tex), run_time=4)
        self.wait(2)

        self.play(H_tex.animate.to_edge(LEFT, buff=0.5), run_time=1)
        self.wait(1)

        tex1 = MathTex(
            r"\boldsymbol{y}_1 = \mathbf{H}\boldsymbol{x}_1 \quad \boldsymbol{y}_2 = \mathbf{H}\boldsymbol{x}_2",
            font_size=36,
        )
        tex2 = MathTex(
            r"\alpha \boldsymbol{y}_1 + \beta \boldsymbol{y}_2 = \mathbf{H}(\alpha \boldsymbol{x}_1 + \beta \boldsymbol{x}_2)",
            font_size=36,
        )
        vgroup = VGroup(tex1, tex2).arrange(DOWN, buff=0.7).to_edge(RIGHT, buff=2)
        self.play(FadeIn(vgroup), run_time=2)
        self.wait(2)

        text = Text(
            "FIR 滤波器是线性时不变系统",
            font_size=32,
        ).next_to(vgroup, DOWN, buff=1)

        self.play(Write(text), run_time=2)
        self.wait(2)


class MatrixFormConvolution3(Scene):
    def construct(self):
        conv_tex = MathTex(
            r"\boldsymbol{y} = \boldsymbol{h} * \boldsymbol{x} = \mathbf{X}\boldsymbol{h}",
        )
        conv_tex.to_edge(UP, buff=0.5)
        self.play(Write(conv_tex), run_time=2)

        text = Text(
            "卷积可交换，也能把数据 \\boldsymbol{x} 展成 Toeplitz 矩阵",
            font_size=32,
        ).next_to(conv_tex, DOWN, buff=0.5)

        self.play(Write(text), run_time=2)
        self.wait(1)

        X_tex = MathTex(
            r"\mathbf{X} = \begin{bmatrix}"
            r"x_1      & 0        & \cdots & 0        & 0        \\"
            r"x_2      & x_1      & \ddots & \vdots   & \vdots   \\"
            r"x_3      & x_2      & \ddots & 0        & 0        \\"
            r"\vdots   & x_3      & \ddots & x_1      & 0        \\"
            r"x_{n-1}  & \vdots   & \ddots & x_2      & x_1      \\"
            r"x_n      & x_{n-1}  & \ddots & \vdots   & x_2      \\"
            r"0        & x_n      & \ddots & x_{n-2}  & \vdots   \\"
            r"0        & 0        & \ddots & x_{n-1}  & x_{n-2}  \\"
            r"\vdots   & \vdots   & \ddots & x_n      & x_{n-1}  \\"
            r"0        & 0        & 0      & \cdots   & x_n"
            r"\end{bmatrix}",
            font_size=26,
        ).next_to(text, DOWN, buff=0.5)

        self.play(Write(X_tex), run_time=4)
        self.wait(2)

        self.play(X_tex.animate.to_edge(LEFT, buff=0.5), run_time=1)
        self.wait(1)

        tex1 = MathTex(
            r"\boldsymbol{y}_1 = \mathbf{X}\boldsymbol{h}_1 \quad \boldsymbol{y}_2 = \mathbf{X}\boldsymbol{h}_2",
            font_size=36,
        )
        tex2 = MathTex(
            r"\alpha \boldsymbol{y}_1 + \beta \boldsymbol{y}_2 = \mathbf{X}(\alpha \boldsymbol{h}_1 + \beta \boldsymbol{h}_2)",
            font_size=36,
        )
        vgroup = VGroup(tex1, tex2).arrange(DOWN, buff=0.7).to_edge(RIGHT, buff=2)
        self.play(FadeIn(vgroup), run_time=2)
        self.wait(2)

        text = Text(
            "交换卷积次序，只是把矩阵由卷积核换成了输入数据",
            font_size=32,
        ).next_to(vgroup, DOWN, buff=1)

        self.play(Write(text), run_time=2)
        self.wait(2)


class MatrixFormConvolution4(Scene):
    def construct(self):
        conv_tex = MathTex(
            r"[1\ 2\ 3\ 4\ 1] * [1\ 2\ 3] = [1 \ 4 \ 10 \ 16 \ 18 \ 14 \ 3]",
            font_size=36,
        )
        conv_tex.to_edge(UP, buff=0.5)
        self.play(Write(conv_tex), run_time=2)

        x_vals = [1, 2, 3, 4, 1]
        h_vals = [1, 2, 3]
        y_vals = np.convolve(h_vals, x_vals)

        def build_kernel_matrix(signal, kernel_len):
            rows = len(signal) + kernel_len - 1
            cols = kernel_len
            data = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    k_idx = r - c
                    row.append(signal[k_idx] if 0 <= k_idx < len(signal) else 0)
                data.append(row)
            return data

        data_matrix = build_kernel_matrix(x_vals, len(h_vals))

        def entry_to_mob(value):
            color = c1 if value != 0 else color_tint(GRAY_B, 0.4)
            return Integer(value, color=color)

        matrix_mob = Matrix(
            data_matrix,
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=entry_to_mob,
        ).scale(0.7)

        h_vector = Matrix(
            [[v] for v in h_vals],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        for entry in h_vector.get_entries():
            entry.set_color(c2)

        result_vector = Matrix(
            [[int(v)] for v in y_vals],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        for entry in result_vector.get_entries():
            entry.set_color(c4)

        times_tex = MathTex(r"\times", font_size=42)
        equals_tex = MathTex(r"=", font_size=42)

        expr_group = VGroup(matrix_mob, times_tex, h_vector, equals_tex, result_vector)
        expr_group.arrange(RIGHT, buff=0.3)
        expr_group.next_to(conv_tex, DOWN, buff=1)

        self.play(Write(matrix_mob), run_time=2)
        self.play(
            FadeIn(times_tex),
            FadeIn(h_vector),
            FadeIn(equals_tex),
            FadeIn(result_vector),
            run_time=1.5,
        )

        matrix_entries = matrix_mob.get_entries()
        rows = len(data_matrix)
        cols = len(h_vals)
        row_groups = [
            VGroup(*matrix_entries[i * cols : (i + 1) * cols]) for i in range(rows)
        ]
        result_entries = result_vector.get_entries()

        row_rect = SurroundingRectangle(row_groups[0], buff=0.15, color=c1)
        input_rect = SurroundingRectangle(h_vector, buff=0.2, color=c2)
        result_rect = SurroundingRectangle(result_entries[0], buff=0.2, color=c4)
        self.play(Create(row_rect), Create(input_rect), Create(result_rect), run_time=1)

        for idx in range(rows):
            current_row = row_groups[idx]
            current_entry = result_entries[idx]
            if idx == 0:
                self.play(
                    Indicate(current_row, color=c1),
                    Indicate(h_vector, color=c2),
                    Indicate(current_entry, color=c4),
                    run_time=1,
                )
            else:
                self.play(
                    row_rect.animate.move_to(current_row),
                    result_rect.animate.move_to(current_entry),
                    run_time=0.8,
                )
                self.play(
                    Indicate(current_row, color=c1),
                    Indicate(current_entry, color=c4),
                    run_time=0.6,
                )

        self.wait(2)

        self.play(
            FadeOut(expr_group),
            FadeOut(row_rect),
            FadeOut(input_rect),
            FadeOut(result_rect),
            run_time=1,
        )

        col1 = matrix_mob = Matrix(
            [[data_matrix[r][0]] for r in range(len(data_matrix))],
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        t1 = MathTex(r"\times", font_size=42)
        h1 = Integer(h_vals[0], color=c2)
        p1 = MathTex(r"+", font_size=42)
        vgroup = VGroup(col1, t1, h1, p1).arrange(RIGHT, buff=0.3)
        vgroup.to_edge(LEFT, buff=2)
        self.play(Write(vgroup), run_time=2)

        col2 = matrix_mob = Matrix(
            [[data_matrix[r][1]] for r in range(len(data_matrix))],
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        t2 = MathTex(r"\times", font_size=42)
        h2 = Integer(h_vals[1], color=c2)
        p2 = MathTex(r"+", font_size=42)
        vgroup2 = VGroup(col2, t2, h2, p2).arrange(RIGHT, buff=0.3)
        vgroup2.next_to(vgroup, RIGHT, buff=0.5)
        self.play(Write(vgroup2), run_time=2)

        col3 = matrix_mob = Matrix(
            [[data_matrix[r][2]] for r in range(len(data_matrix))],
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        t3 = MathTex(r"\times", font_size=42)
        h3 = Integer(h_vals[2], color=c2)
        vgroup3 = VGroup(col3, t3, h3).arrange(RIGHT, buff=0.3)
        vgroup3.next_to(vgroup2, RIGHT, buff=0.5)
        self.play(Write(vgroup3), run_time=2)

        eq = MathTex(r"=", font_size=42)
        result = Matrix(
            [[int(v)] for v in y_vals],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        for entry in result.get_entries():
            entry.set_color(c4)
        result_group = VGroup(eq, result).arrange(RIGHT, buff=0.3)
        result_group.next_to(vgroup3, RIGHT, buff=0.5)
        self.play(Write(result_group), run_time=2)

        col1_rotate = col1.copy()
        col2_rotate = col2.copy()
        col3_rotate = col3.copy()

        col1_rotate.rotate(90 * DEGREES).next_to(conv_tex, DOWN, buff=0.5)
        col2_rotate.rotate(90 * DEGREES).next_to(col1_rotate, DOWN, buff=0.5)
        col3_rotate.rotate(90 * DEGREES).next_to(col2_rotate, DOWN, buff=0.5)

        t1_copy = t1.copy()
        t2_copy = t2.copy()
        t3_copy = t3.copy()

        t1_copy.next_to(col1_rotate, RIGHT, buff=0.3)
        t2_copy.next_to(col2_rotate, RIGHT, buff=0.3)
        t3_copy.next_to(col3_rotate, RIGHT, buff=0.3)

        h1_copy = h1.copy()
        h2_copy = h2.copy()
        h3_copy = h3.copy()

        h1_copy.next_to(t1_copy, RIGHT, buff=0.3)
        h2_copy.next_to(t2_copy, RIGHT, buff=0.3)
        h3_copy.next_to(t3_copy, RIGHT, buff=0.3)

        vgroup1 = VGroup(col1_rotate, t1_copy, h1_copy)
        vgroup2 = VGroup(col2_rotate, t2_copy, h2_copy)
        vgroup3 = VGroup(col3_rotate, t3_copy, h3_copy)

        vgroup3.next_to(conv_tex, DOWN, buff=0.5)
        vgroup2.next_to(vgroup3, DOWN, buff=0.5)
        vgroup1.next_to(vgroup2, DOWN, buff=0.5)

        result_copy = result.copy()
        result_copy.rotate(90 * DEGREES)
        result_copy.next_to(vgroup1, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(
            FadeOut(eq),
            FadeOut(p1),
            FadeOut(p2),
            Transform(col1, col1_rotate),
            Transform(col2, col2_rotate),
            Transform(col3, col3_rotate),
            Transform(t1, t1_copy),
            Transform(t2, t2_copy),
            Transform(t3, t3_copy),
            Transform(h1, h1_copy),
            Transform(h2, h2_copy),
            Transform(h3, h3_copy),
            Transform(result, result_copy),
            run_time=3,
        )

        row1 = Matrix(
            [[data_matrix[r][2] * 3 for r in range(len(data_matrix))]],
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        bra = row1.get_brackets()
        row1.remove(bra[0], bra[1])
        row1.move_to(vgroup3.get_center())

        row2 = Matrix(
            [[data_matrix[r][1] * 2 for r in range(len(data_matrix))]],
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        bra = row2.get_brackets()
        row2.remove(bra[0], bra[1])
        row2.move_to(vgroup2.get_center())

        row3 = Matrix(
            [[data_matrix[r][0] * 1 for r in range(len(data_matrix))]],
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        bra = row3.get_brackets()
        row3.remove(bra[0], bra[1])
        row3.move_to(vgroup1.get_center())

        row4 = Matrix(
            [[y_vals[r] for r in range(len(y_vals))]],
            element_to_mobject=entry_to_mob,
        ).scale(0.7)
        bra = row4.get_brackets()
        row4.remove(bra[0], bra[1])
        row4.next_to(row3, DOWN, buff=1)

        self.play(
            FadeOut(col1),
            FadeOut(t1),
            FadeOut(h1),
            FadeOut(col2),
            FadeOut(t2),
            FadeOut(h2),
            FadeOut(col3),
            FadeOut(t3),
            FadeOut(h3),
            Transform(vgroup3, row1),
            Transform(vgroup2, row2),
            Transform(vgroup1, row3),
            Transform(result, row4),
            run_time=2,
        )
        self.wait(2)

        self.play(
            vgroup1.animate.next_to(result, UP, 0.5),
            run_time=0.3,
        )
        self.play(
            vgroup2.animate.next_to(vgroup1, UP, 0.5),
            run_time=0.3,
        )
        self.play(
            vgroup3.animate.next_to(vgroup2, UP, 0.5),
            run_time=0.3,
        )

        line2 = Line(
            vgroup1.get_bottom() + 0.2 * DOWN + 3 * LEFT,
            vgroup1.get_bottom() + 0.2 * DOWN + 3 * RIGHT,
            color=WHITE,
            stroke_width=2,
        )
        line1 = Line(
            vgroup3.get_top() + 0.2 * UP + 3 * LEFT,
            vgroup3.get_top() + 0.2 * UP + 3 * RIGHT,
            color=WHITE,
            stroke_width=2,
        )

        vec1 = Matrix(
            [[1, 2, 3]],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        bra = vec1.get_brackets()
        vec1.remove(bra[0], bra[1])
        vec1.next_to(vgroup3, UP, buff=0.5, aligned_edge=RIGHT)

        vec2 = Matrix(
            [[1, 2, 3, 4, 1]],
            left_bracket="(",
            right_bracket=")",
        ).scale(0.7)
        bra = vec2.get_brackets()
        vec2.remove(bra[0], bra[1])
        vec2.next_to(vec1, UP, buff=0.5, aligned_edge=RIGHT)

        self.play(Create(line1), Create(line2), Create(vec1), Create(vec2), run_time=1)
