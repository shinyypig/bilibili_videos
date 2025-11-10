from manim import *
from utils import *
import numpy as np


class DiagScene(Scene):
    def construct(self):
        def jordan_block_2x2(lmbd):
            return np.array([[lmbd, 0], [0, 1]])

        logo = Logo()
        self.add(logo)

        title = Text("情形2：可对角化", font_size=38, color=WHITE)
        title.to_edge(UL, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)

        # --- 2D Jordan block section ---
        matrix_2d_1 = MathTex(
            r"\mathbf{D} = ",
            font_size=40,
            color=WHITE,
        )
        matrix_2d_2 = MathTex(
            r"\begin{bmatrix} \lambda & 0 \\ 0 & 1 \end{bmatrix}",
            font_size=40,
            color=WHITE,
        )
        matrix_2d_2.next_to(matrix_2d_1, RIGHT, buff=0.3)
        matrix_2d = VGroup(matrix_2d_1, matrix_2d_2)

        matrix_2d.to_edge(LEFT, buff=1.0).shift(UP * 0.8)

        self.play(FadeIn(matrix_2d, shift=RIGHT))
        self.wait(0.4)

        axes = (
            NumberPlane(
                x_range=[-3, 3],
                y_range=[-3, 3],
                x_length=6,
                y_length=6,
            )
            .add_coordinates()
            .shift(RIGHT * 1.5)
        )
        self.play(FadeIn(axes))

        square = Square(side_length=1, color=c1, fill_opacity=0.4)
        square.move_to(axes.c2p(0.5, 0.5))  # 放在第一象限
        self.play(FadeIn(square))

        lambda_tracker_2d = ValueTracker(2.0)
        lambda_label_2d = MathTex(r"\lambda =", font_size=32, color=WHITE)
        lambda_value_2d = DecimalNumber(
            lambda_tracker_2d.get_value(),
            num_decimal_places=2,
            font_size=32,
            color=WHITE,
        )
        lambda_value_2d.add_updater(
            lambda dec: dec.set_value(lambda_tracker_2d.get_value())
        )
        lambda_group_2d = VGroup(lambda_label_2d, lambda_value_2d).arrange(
            RIGHT, buff=0.2
        )
        lambda_group_2d.next_to(matrix_2d, DOWN, buff=0.6, aligned_edge=LEFT)
        self.play(Write(lambda_group_2d))

        lambda_number_line = NumberLine(
            x_range=[-2, 2, 1],
            length=4,
            include_numbers=True,
            include_tip=False,
            font_size=24,
            color=WHITE,
        )
        lambda_number_line.next_to(
            lambda_group_2d, DOWN, buff=0.5, aligned_edge=LEFT
        ).shift(LEFT * 0.4)
        self.play(Create(lambda_number_line))

        lambda_dot = Dot(color=c2, radius=0.06)
        lambda_dot.add_updater(
            lambda dot: dot.move_to(
                lambda_number_line.n2p(lambda_tracker_2d.get_value())
            )
        )
        self.play(FadeIn(lambda_dot))

        square_trans = always_redraw(
            lambda: square.copy()
            .set_color(c2)
            .apply_matrix(
                jordan_block_2x2(lambda_tracker_2d.get_value()),
                about_point=axes.c2p(0, 0),
            )
        )

        self.play(FadeIn(square_trans))
        self.play(lambda_tracker_2d.animate.set_value(-2), run_time=8, rate_func=linear)

        self.wait(0.3)

        label1 = Text("缩放", font_size=28, color=WHITE)
        label2 = Text("投影", font_size=28, color=WHITE)
        label3 = Text("反射", font_size=28, color=WHITE)

        vgroup = VGroup(label3, label2, label1).arrange(
            RIGHT, aligned_edge=LEFT, buff=1
        )

        vgroup.next_to(lambda_number_line, DOWN, buff=1.0)

        self.play(Write(label1))
        self.play(Write(label2))
        self.play(Write(label3))

        # clear page
        self.play(
            FadeOut(vgroup),
            FadeOut(square),
            FadeOut(axes),
            FadeOut(lambda_dot),
            FadeOut(lambda_number_line),
            FadeOut(lambda_group_2d),
            FadeOut(matrix_2d),
            FadeOut(square_trans),
        )

        self.wait(0.3)

        # rotation (complex eigenvalue) section
        tex1 = MathTex(
            r"\mathbf{T} \boldsymbol{v} = \lambda \boldsymbol{v}",
            font_size=40,
            color=WHITE,
        )
        tex1.to_edge(UP, buff=1.5)
        self.play(Write(tex1))

        tex2 = MathTex(
            r"\mathbf{T} \overline{\boldsymbol{v}} = \overline{\mathbf{T}} \overline{\boldsymbol{v}} = \overline{\mathbf{T} \boldsymbol{v}} = \overline{\lambda \boldsymbol{v}} = \overline{\lambda} \overline{\boldsymbol{v}}",
            font_size=36,
            color=WHITE,
        )
        tex2.next_to(tex1, DOWN, buff=0.5)
        self.play(Write(tex2))
        self.wait(1)

        self.play(FadeOut(VGroup(tex1, tex2), shift=UP))

        tex5 = MathTex(
            r"\boldsymbol{v} = \boldsymbol{a} + i \boldsymbol{b}, \quad \lambda = r e^{i\theta}",
            font_size=36,
            color=WHITE,
        )
        tex5.to_edge(UP, buff=1.5)
        self.play(Write(tex5))

        tex3 = MathTex(
            r"\mathbf{T}  = \begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix} \begin{bmatrix} \lambda & 0 \\ 0 & \overline{\lambda} \end{bmatrix} \begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix}^{-1}",
            font_size=36,
            color=WHITE,
        )
        tex3.next_to(tex5, DOWN, buff=0.5)
        self.play(Write(tex3))
        self.wait(1)

        tex4_1 = MathTex(
            r"\mathbf{T}  = ",
            font_size=36,
            color=WHITE,
        )
        tex4_2 = MathTex(
            r"\begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix}",
            font_size=36,
            color=c1,
        )
        tex4_3 = MathTex(
            r"\begin{bmatrix} 1 & 1 \\ i & -i \end{bmatrix}^{-1}",
            font_size=36,
            color=c1,
        )
        tex4_4 = MathTex(
            r"\begin{bmatrix} 1 & 1 \\ i & -i \end{bmatrix}", font_size=36, color=c2
        )
        tex4_5 = MathTex(
            r" \begin{bmatrix} \lambda & 0 \\ 0 & \overline{\lambda} \end{bmatrix}",
            font_size=36,
            color=c2,
        )
        tex4_6 = MathTex(
            r"\begin{bmatrix} 1 & 1 \\ i & -i \end{bmatrix}^{-1}",
            font_size=36,
            color=c2,
        )
        tex4_7 = MathTex(
            r"\begin{bmatrix} 1 & 1 \\ i & -i \end{bmatrix}",
            font_size=36,
            color=c1,
        )
        tex4_8 = MathTex(
            r"\begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix}^{-1}",
            font_size=36,
            color=c1,
        )

        vgrp = VGroup(
            tex4_1, tex4_2, tex4_3, tex4_4, tex4_5, tex4_6, tex4_7, tex4_8
        ).arrange(RIGHT, buff=0.3)

        vgrp.next_to(tex3, DOWN, buff=0.5)
        self.play(Write(tex4_1), Write(tex4_2), Write(tex4_5), Write(tex4_8))
        self.wait(1)
        self.play(Write(tex4_3), Write(tex4_4), Write(tex4_6), Write(tex4_7))
        self.wait(1)

        box1 = SurroundingRectangle(VGroup(tex4_3, tex4_4), buff=0.1, color=WHITE)
        box2 = SurroundingRectangle(VGroup(tex4_6, tex4_7), buff=0.1, color=WHITE)
        self.play(Create(box1), Create(box2))

        self.wait(1)

        tex6_1 = MathTex(
            r"\mathbf{T}  = ",
            font_size=36,
            color=WHITE,
        )
        tex6_2 = MathTex(
            r"\begin{bmatrix} \boldsymbol{a} & \boldsymbol{b} \end{bmatrix}",
            font_size=36,
            color=c1,
        )
        tex6_3 = MathTex(
            r"r\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}",
            font_size=36,
            color=c2,
        )
        tex6_4 = MathTex(
            r"\begin{bmatrix} \boldsymbol{a} & \boldsymbol{b} \end{bmatrix}^{-1}",
            font_size=36,
            color=c1,
        )

        vgrp2 = VGroup(tex6_1, tex6_2, tex6_3, tex6_4).arrange(RIGHT, buff=0.3)

        vgrp2.next_to(vgrp, DOWN, buff=0.5)

        self.play(Write(tex6_1), run_time=1)
        gp1 = VGroup(tex4_2, tex4_3).copy()
        gp2 = VGroup(tex4_4, tex4_5, tex4_6).copy()
        gp3 = VGroup(tex4_7, tex4_8).copy()
        self.play(
            Transform(gp1, tex6_2),
            Transform(gp3, tex6_4),
            run_time=2,
        )
        self.play(Transform(gp2, tex6_3), run_time=2)
        self.wait(2)

        # clear page
        self.play(
            FadeOut(tex5),
            FadeOut(tex3),
            FadeOut(vgrp),
            FadeOut(box1),
            FadeOut(box2),
            FadeOut(vgrp2),
            FadeOut(gp1),
            FadeOut(gp2),
            FadeOut(gp3),
        )
        self.wait(0.3)

        matrix_2d = MathTex(
            r"r\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}",
            font_size=40,
            color=WHITE,
        )

        matrix_2d.to_edge(LEFT, buff=1.0).shift(UP * 2)

        self.play(FadeIn(matrix_2d, shift=RIGHT))
        self.wait(0.4)

        r_tracker = ValueTracker(1.0)
        theta_tracker = ValueTracker(0.0)

        r_tex = MathTex(r"r = ", font_size=40, color=WHITE)
        r_value = DecimalNumber(
            r_tracker.get_value(),
            num_decimal_places=2,
            font_size=40,
            color=WHITE,
        )
        theta_tex = MathTex(r"\quad \theta = ", font_size=40, color=WHITE)
        theta_value = DecimalNumber(
            theta_tracker.get_value(),
            num_decimal_places=2,
            font_size=40,
            color=WHITE,
        )
        theta_unit = MathTex(r"\pi", font_size=40, color=WHITE)

        r_label = VGroup(r_tex, r_value).arrange(RIGHT, buff=0.1)
        theta_label = VGroup(theta_tex, theta_value, theta_unit).arrange(
            RIGHT, buff=0.1
        )
        label = VGroup(r_label, theta_label).arrange(RIGHT, buff=0.3)

        label.next_to(matrix_2d, DOWN, buff=0.6, aligned_edge=LEFT)
        theta_value.add_updater(lambda m: m.set_value(theta_tracker.get_value()))
        r_value.add_updater(lambda m: m.set_value(r_tracker.get_value()))
        self.play(Write(label))

        axes_complex = (
            NumberPlane(
                x_range=[-2, 2],
                y_range=[-2, 2],
                x_length=3,
                y_length=3,
            )
            .add_coordinates()
            .next_to(label, DOWN, buff=0.5)
            .shift(RIGHT * 0.4)
        )
        self.play(Write(axes_complex))

        circle = Circle(radius=0.75, color=GREY, stroke_width=1)
        circle.move_to(axes_complex.c2p(0, 0))
        self.play(Write(circle))

        point = Dot(color=c2, radius=0.06)
        point.add_updater(
            lambda p: p.move_to(
                axes_complex.c2p(
                    r_tracker.get_value() * np.cos(theta_tracker.get_value() * np.pi),
                    r_tracker.get_value() * np.sin(theta_tracker.get_value() * np.pi),
                )
            )
        )
        self.play(FadeIn(point))

        axes = (
            NumberPlane(
                x_range=[-3, 3],
                y_range=[-3, 3],
                x_length=6,
                y_length=6,
            )
            .add_coordinates()
            .shift(RIGHT * 1.5)
        )
        self.play(FadeIn(axes))

        square = Square(side_length=1, color=c1, fill_opacity=0.4)
        square.move_to(axes.c2p(0.5, 0.5))  # 放在第一象限
        self.play(FadeIn(square))

        def rotate_matrix(r, theta):
            return r * np.array(
                [
                    [np.cos(theta * np.pi), np.sin(theta * np.pi)],
                    [-np.sin(theta * np.pi), np.cos(theta * np.pi)],
                ]
            )

        square_trans = always_redraw(
            lambda: square.copy()
            .set_color(c2)
            .apply_matrix(
                rotate_matrix(r_tracker.get_value(), theta_tracker.get_value()),
                about_point=axes.c2p(0, 0),
            )
        )

        self.play(FadeIn(square_trans))

        self.play(theta_tracker.animate.set_value(2), run_time=4)
        self.play(theta_tracker.animate.set_value(0.25), run_time=4)
        self.play(r_tracker.animate.set_value(2), run_time=3)
        self.play(r_tracker.animate.set_value(0.5), run_time=3)
