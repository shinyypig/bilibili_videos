from manim import *
from utils import *
import numpy as np


class NonDiagScene(ThreeDScene):
    def construct(self):
        def jordan_block_2x2(lmbd):
            return np.array([[1, 1 / lmbd], [0, 1]])

        logo = Logo()
        self.add(logo)

        title = Text("情形1：不可对角化", font_size=38, color=WHITE)
        title.to_edge(UL, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)
        self.add_fixed_in_frame_mobjects(logo, title)

        # --- 2D Jordan block section ---
        matrix_2d_1 = MathTex(
            r"\mathbf{J} = \lambda ",
            font_size=40,
            color=WHITE,
        )
        matrix_2d_2 = MathTex(
            r"\begin{bmatrix} 1 & \frac{1}{\lambda} \\ 0 & 1 \end{bmatrix}",
            font_size=40,
            color=WHITE,
        )
        matrix_2d_2.next_to(matrix_2d_1, RIGHT, buff=0.3)
        matrix_2d = VGroup(matrix_2d_1, matrix_2d_2)

        matrix_2d.to_edge(LEFT, buff=1.0).shift(UP * 0.8)

        self.play(FadeIn(matrix_2d, shift=RIGHT))
        self.wait(0.4)

        # 添加一个框将矩阵括起来
        box = SurroundingRectangle(matrix_2d_2, buff=0.2, color=c1)
        self.play(FadeIn(box, shift=UP))

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

        lambda_tracker_2d = ValueTracker(1.0)
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

        square_trans = always_redraw(
            lambda: square.copy()
            .set_color(c2)
            .apply_matrix(
                jordan_block_2x2(lambda_tracker_2d.get_value()),
                about_point=axes.c2p(0, 0),
            )
        )

        self.play(FadeIn(square_trans))
        self.play(lambda_tracker_2d.animate.set_value(2), run_time=2)
        self.play(lambda_tracker_2d.animate.set_value(0.5), run_time=2.2)

        lambda_value_2d.clear_updaters()
        square_trans.suspend_updating()

        self.play(
            FadeOut(VGroup(square_trans, square, axes), shift=DOWN),
            FadeOut(VGroup(matrix_2d, lambda_group_2d, box), shift=LEFT),
        )
        self.wait(0.3)

        matrix_3d_1 = MathTex(
            r"\mathbf{J} = \lambda ",
            font_size=40,
            color=WHITE,
        )
        matrix_3d_2 = MathTex(
            r"\begin{bmatrix} 1 & \frac{1}{\lambda} & 0 \\ 0 & 1 & \frac{1}{\lambda} \\ 0 & 0 & 1 \end{bmatrix}",
            font_size=40,
            color=WHITE,
        )
        matrix_3d_2.next_to(matrix_3d_1, RIGHT, buff=0.3)
        matrix_3d = VGroup(matrix_3d_1, matrix_3d_2)
        matrix_3d.to_edge(LEFT, buff=1.0).shift(UP * 0.8)

        box_3d = SurroundingRectangle(matrix_3d_2, buff=0.2, color=c1)

        self.play(FadeIn(matrix_3d, shift=RIGHT))
        self.play(Create(box_3d))
        self.add_fixed_in_frame_mobjects(matrix_3d, box_3d)

        lambda_tracker_3d = ValueTracker(1.0)
        lambda_label_3d = MathTex(r"\lambda =", font_size=32, color=WHITE)
        lambda_value_3d = DecimalNumber(
            lambda_tracker_3d.get_value(),
            num_decimal_places=2,
            font_size=32,
            color=WHITE,
        )
        lambda_value_3d.add_updater(
            lambda dec: dec.set_value(lambda_tracker_3d.get_value())
        )
        lambda_group_3d = VGroup(lambda_label_3d, lambda_value_3d).arrange(
            RIGHT, buff=0.2
        )

        lambda_group_3d.next_to(matrix_3d, DOWN, buff=0.6, aligned_edge=LEFT)
        self.play(Write(lambda_group_3d))

        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        axes_3d.shift(RIGHT * 1.5)
        axes_3d.rotate(
            PI / 9,
            axis=RIGHT,
            about_point=axes_3d.c2p(0, 0, 0),
        )
        self.play(FadeIn(axes_3d))

        cube = Cube(
            side_length=1,
            fill_color=c1,
            fill_opacity=0.25,
            stroke_color=c1,
            stroke_width=1.5,
        )
        cube.rotate(PI / 9, axis=RIGHT, about_point=axes_3d.c2p(0, 0, 0))
        cube.shift(axes_3d.c2p(0.5, 0.5, 0.5))
        self.play(FadeIn(cube))

        # 动态获取“局部 z 轴”（即 axes_3d 的当前 z 轴方向）
        def local_z_axis_dir():
            a, b = axes_3d.y_axis.get_start_and_end()
            return normalize(b - a)

        # def jordan_block_3x3(lmbd):
        #     return np.array(
        #         [
        #             [1, 1 / lmbd, 0],
        #             [0, 1, 1 / lmbd],
        #             [0, 0, 1],
        #         ]
        #     )

        def jordan_block_3x3_local(lmbd: float):
            # 防止除零
            eps = 1e-6
            if abs(lmbd) < eps:
                lmbd = eps if lmbd >= 0 else -eps
            return np.array(
                [
                    [1.0, 0.0, 1.0 / lmbd],
                    [0.0, 1.0, 0],
                    [0.0, 1.0 / lmbd, 1.0],
                ],
                dtype=float,
            )

        def axes_basis_matrix(axes: ThreeDAxes):
            # 取当前三轴的世界坐标方向（包含尺度），组列向量为基底矩阵 B
            x0, x1 = axes.x_axis.get_start_and_end()
            y0, y1 = axes.y_axis.get_start_and_end()
            z0, z1 = axes.z_axis.get_start_and_end()
            X = x1 - x0
            Y = y1 - y0
            Z = z1 - z0
            return np.column_stack([X, Y, Z])  # 3x3

        def local_shear_as_world_matrix(axes: ThreeDAxes, lmbd: float):
            M_local = jordan_block_3x3_local(lmbd)  # 在 axes_3d 的局部坐标
            B = axes_basis_matrix(axes)  # 局部->世界 的基底
            Binv = np.linalg.inv(B)
            return B @ M_local @ Binv  # 共轭到世界坐标

        cube_trans = always_redraw(
            lambda: cube.copy()
            .set_color(c2)
            .set_fill(c2, opacity=0.4)
            .apply_matrix(
                local_shear_as_world_matrix(axes_3d, lambda_tracker_3d.get_value()),
                about_point=axes_3d.c2p(0, 0, 0),
            )
        )
        self.play(FadeIn(cube_trans))

        grp = VGroup(axes_3d, cube, cube_trans)
        grp.add_updater(
            lambda m, dt: m.rotate(
                0.6 * dt,
                axis=local_z_axis_dir(),
                about_point=axes_3d.c2p(0, 0, 0),
            )
        )

        self.add(grp)
        self.play(lambda_tracker_3d.animate.set_value(10), run_time=8)
        self.play(lambda_tracker_3d.animate.set_value(1), run_time=8)

        # 清理场景
        # lambda_value_3d.clear_updaters()
        # cube_trans.suspend_updating()
        # grp.suspend_updating()
        # self.play(
        #     FadeOut(grp, shift=DOWN),
        #     FadeOut(VGroup(matrix_3d, lambda_group_3d, box_3d), shift=LEFT),
        # )
        # self.wait(0.3)
