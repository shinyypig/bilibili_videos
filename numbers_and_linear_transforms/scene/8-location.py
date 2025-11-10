from manim import *
from utils import *
import numpy as np


class LocationScene(ThreeDScene):
    def construct(self):
        logo = Logo()
        self.add(logo)

        title = Text("线性变换的场地", font_size=38, color=WHITE)
        title.to_edge(UL, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)

        text1 = Text("给定一个线性变换", font_size=32, color=WHITE)
        tex1 = MathTex(
            r"\mathbf{T} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1}", font_size=48
        )
        grp = VGroup(text1, tex1).arrange(DOWN, buff=0.3)
        grp.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(Write(grp))
        self.wait(0.3)

        tex2 = MathTex(
            r"\mathbf{J}",
            font_size=36,
        )
        text2 = Text("：线性变换的类型", font_size=32, color=WHITE)
        grp2 = VGroup(tex2, text2).arrange(RIGHT, buff=0.3)
        grp2.next_to(grp, DOWN, buff=0.8).align_to(grp, LEFT)
        self.play(Write(grp2))
        self.wait(0.3)

        tex3 = MathTex(
            r"\mathbf{P}",
            font_size=36,
        )
        text3 = Text("：线性变换的场地", font_size=32, color=WHITE)
        grp3 = VGroup(tex3, text3).arrange(RIGHT, buff=0.3)
        grp3.next_to(grp2, DOWN, buff=0.2).align_to(grp2, LEFT)
        self.play(Write(grp3))
        self.wait(0.3)

        tex4 = MathTex(
            r"\begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{bmatrix} \begin{bmatrix} 1.5 & 0 \\ 0 & 0.5 \end{bmatrix} \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{bmatrix}^{-1}",
            font_size=18,
        )
        tex4.next_to(grp3, DOWN, buff=0.8).align_to(grp3, LEFT)
        self.play(Write(tex4))

        axes = NumberPlane(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=6,
            y_length=6,
        ).add_coordinates()
        axes.to_edge(RIGHT, buff=2)  # 关键：整体靠左

        self.play(FadeIn(axes))

        square = Square(side_length=1, color=c1, fill_opacity=0.4)
        square.move_to(axes.c2p(0, 0, 0))  # 依然用 axes 坐标

        self.play(FadeIn(square))

        mat = np.array([[1, 0.5], [0.5, 1]])
        P = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        D = np.array([[1.5, 0], [0, 0.5]])
        Pinv = P.T

        square_copy = square.copy()
        square_copy.set_color(c2)

        square_copy2 = square.copy()
        square_copy2.set_color(c3)

        self.play(
            ApplyMatrix(mat, square_copy, axes.c2p(0, 0)),
            run_time=2,
        )

        self.play(
            # axes.animate.set_color(GRAY).set_opacity(0.4),
            ApplyMatrix(Pinv, square_copy2, axes.c2p(0, 0)),
            run_time=2,
        )
        self.play(
            ApplyMatrix(D, square_copy2, axes.c2p(0, 0)),
            run_time=2,
        )
        self.play(
            ApplyMatrix(P, square_copy2, axes.c2p(0, 0)),
            run_time=2,
        )

        self.play(
            FadeOut(
                text1,
                tex2,
                text2,
                tex3,
                text3,
                tex4,
                grp,
                grp2,
                grp3,
                # axes,
                square,
                square_copy,
                square_copy2,
            ),
        )
        self.wait(0.3)

        def Pmat(theta):
            return np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )

        Pmat_label = MathTex(
            r"\mathbf{P} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )
        Pmat_label.to_edge(UP, buff=2).to_edge(LEFT, buff=1)
        self.play(Write(Pmat_label))

        Dmat_label = MathTex(
            r"\mathbf{D} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )
        Dmat_label.next_to(Pmat_label, DOWN, buff=0.5).align_to(Pmat_label, LEFT)
        self.play(Write(Dmat_label))
        self.wait(0.3)

        T_label = MathTex(
            r"\mathbf{T} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}",
            font_size=32,
            color=WHITE,
        )
        T_label.next_to(Dmat_label, DOWN, buff=0.5).align_to(Dmat_label, LEFT)
        self.play(Write(T_label))
        self.wait(0.3)

        Dmat = np.array([[2, 0], [0, 1]])
        theta_tracker = ValueTracker(0)
        theta_label = MathTex(r"\theta =", font_size=32, color=WHITE)
        theta_value = DecimalNumber(
            theta_tracker.get_value() / np.pi,
            num_decimal_places=2,
            font_size=32,
            color=WHITE,
        )
        theta_value.add_updater(
            lambda dec: dec.set_value(theta_tracker.get_value() / np.pi)
        )
        theta_unit = MathTex(r"\pi", font_size=32, color=WHITE)
        theta_group = VGroup(theta_label, theta_value, theta_unit).arrange(
            RIGHT, buff=0.2
        )
        theta_group.next_to(T_label, DOWN, buff=0.6, aligned_edge=LEFT)
        self.play(Write(theta_group))

        circle = Circle(radius=1, color=c1, stroke_opacity=1)
        circle.move_to(axes.c2p(0, 0, 0))
        self.play(FadeIn(circle))

        circle_trans = always_redraw(
            lambda: circle.copy()
            .set_color(c2)
            .apply_matrix(
                Pmat(theta_tracker.get_value())
                @ Dmat
                @ np.linalg.inv(Pmat(theta_tracker.get_value())),
                about_point=axes.c2p(0, 0),
            )
        )

        arrow1 = always_redraw(
            lambda: Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(
                    2 * np.cos(theta_tracker.get_value()),
                    2 * np.sin(theta_tracker.get_value()),
                ),
                buff=0,
            ).set_color(WHITE)
        )
        arrow2 = always_redraw(
            lambda: Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(
                    -np.sin(theta_tracker.get_value()),
                    np.cos(theta_tracker.get_value()),
                ),
                buff=0,
            ).set_color(WHITE)
        )
        self.play(FadeIn(circle_trans), FadeIn(arrow1), FadeIn(arrow2))
        self.play(theta_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        self.wait(1)

        self.play(
            FadeOut(
                Pmat_label,
                Dmat_label,
                T_label,
                theta_group,
                circle,
                circle_trans,
                arrow1,
                arrow2,
                axes,
            ),
        )

        tex1 = MathTex(
            r"\mathbf{T} = \begin{bmatrix} \boldsymbol{v_1} & \boldsymbol{v_2}  & \cdots  & \boldsymbol{v_n} \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix} \begin{bmatrix} \boldsymbol{v_1} & \boldsymbol{v_2}  & \cdots  & \boldsymbol{v_n} \end{bmatrix}^{-1}",
            font_size=32,
        )
        tex1.to_edge(UP, buff=1)
        self.play(Write(tex1))
        self.wait(0.3)

        tex2 = MathTex(
            r"\lambda_i \in \mathbb{R}, \boldsymbol{v_i} \in \mathbb{R}^n",
            font_size=36,
        )

        text2 = Text(
            "特征值表明线性变换的类型，特征向量表明变换的场地",
            font_size=32,
            color=WHITE,
        )
        grp2 = VGroup(tex2, text2).arrange(DOWN, buff=0.3)
        grp2.next_to(tex1, DOWN, buff=0.5)
        self.play(Write(grp2))
        self.wait(0.3)

        tex3 = MathTex(
            r"\mathbf{J}_{\lambda_k} = \begin{bmatrix} \lambda_k & 1 & 0 & \cdots & 0 \\ 0 & \lambda_k & 1 & \cdots & 0 \\ 0 & 0 & \lambda_k & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & 1 \\ 0 & 0 & 0 & \cdots & \lambda_k \end{bmatrix}, \quad \mathbf{V}_{\lambda_k} = \begin{bmatrix} \boldsymbol{v_{k_1}} & \boldsymbol{v_{k_2}} & \cdots & \boldsymbol{v_{k_m}} \end{bmatrix}",
            font_size=32,
        )
        tex3.next_to(grp2, DOWN, buff=0.8)
        self.play(Write(tex3))
        self.wait(0.3)

        self.play(FadeOut(tex1, grp2, tex3))

        # rotation (complex eigenvalue) section
        tex1 = MathTex(
            r"\begin{split} \mathbf{T}  & = \begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix} \begin{bmatrix} r e^{i\theta} & 0 \\ 0 & r e^{-i\theta} \end{bmatrix} \begin{bmatrix} \boldsymbol{v} & \overline{\boldsymbol{v}} \end{bmatrix}^{-1} \\ & = \begin{bmatrix} \boldsymbol{a} & \boldsymbol{b} \end{bmatrix}  r\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \boldsymbol{a} & \boldsymbol{b} \end{bmatrix}^{-1} \end{split}",
            font_size=36,
            color=WHITE,
        )

        tex1.to_edge(UP, buff=1).to_edge(LEFT, buff=0.5)
        self.play(Write(tex1))
        self.wait(0.3)

        self.play(tex1.animate.scale(0.6, about_edge=UL))

        tex2 = MathTex(
            r"\mathbf{T} = \begin{bmatrix} -0.2287 & -0.5854 & -0.7778 \\ 0.7256 & -0.6352 & 0.2646 \\ -0.6489 & -0.5039 & 0.5701 \end{bmatrix}",
            font_size=28,
            color=WHITE,
        )
        tex2.next_to(tex1, DOWN, buff=0.5).align_to(tex1, LEFT)
        self.play(Write(tex2))
        self.wait(0.3)

        tex3 = MathTex(
            r"\mathbf{P} = \begin{bmatrix} -0.50  & -0.03 + 0.61i & -0.03 - 0.61i \\ -0.08  & 0.70 & 0.70 \\ 0.86 & 0.05 + 0.36i & 0.05 - 0.36i \end{bmatrix}",
            font_size=28,
            color=WHITE,
        )
        tex3.next_to(tex2, DOWN, buff=0.5).align_to(tex2, LEFT)
        self.play(Write(tex3))
        self.wait(0.3)

        tex4 = MathTex(
            r"\mathbf{D} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -0.6469 + 0.7626i & 0 \\ 0 & 0 & -0.6469 - 0.7626i \end{bmatrix}",
            font_size=28,
            color=WHITE,
        )

        tex4.next_to(tex3, DOWN, buff=0.5).align_to(tex3, LEFT)
        self.play(Write(tex4))
        self.wait(0.3)

        vgrp = VGroup(tex2, tex3, tex4)

        self.play(vgrp.animate.scale(0.6, about_edge=UL))
        self.wait(0.3)

        tex5 = MathTex(
            r"\boldsymbol{a} = \begin{bmatrix} -0.03 \\ 0.70 \\ 0.05 \end{bmatrix}",
            font_size=28,
            color=c3,
        )
        tex6 = MathTex(
            r"\boldsymbol{b} = \begin{bmatrix} 0.61 \\ 0 \\ 0.36 \end{bmatrix}",
            font_size=28,
            color=c4,
        )
        vbgrp = VGroup(tex5, tex6).arrange(RIGHT, buff=0.5)
        vbgrp.next_to(vgrp, DOWN, buff=0.5).align_to(vgrp, LEFT)
        self.play(Write(vbgrp))
        self.wait(0.3)

        tex7 = MathTex(
            r"r e^{i\theta} = -0.6469 + 0.7626i = 1 e^{i \pi \times 0.7239}",
            font_size=28,
            color=WHITE,
        )
        tex7.next_to(vbgrp, DOWN, buff=0.5).align_to(vbgrp, LEFT)
        self.play(Write(tex7))
        self.wait(0.3)

        self.add_fixed_in_frame_mobjects(
            logo,
            title,
            tex1,
            tex2,
            tex3,
            tex4,
            vbgrp,
            tex7,
        )

        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        axes_3d.to_edge(RIGHT, buff=2)
        axes_3d.rotate(
            -PI / 2,
            axis=RIGHT,
            about_point=axes_3d.c2p(0, 0, 0),
        ).rotate(
            -3 * PI / 4,
            axis=UP,
            about_point=axes_3d.c2p(0, 0, 0),
        ).rotate(
            PI / 6,
            axis=RIGHT,
            about_point=axes_3d.c2p(0, 0, 0),
        )

        self.play(Create(axes_3d))
        self.wait(0.3)

        cube = (
            Cube(
                side_length=1,
                fill_color=c1,
                fill_opacity=0.1,
                stroke_color=c1,
                stroke_width=1.5,
            )
            .rotate(
                -PI / 2,
                axis=RIGHT,
                about_point=axes_3d.c2p(0, 0, 0),
            )
            .rotate(
                -3 * PI / 4,
                axis=UP,
                about_point=axes_3d.c2p(0, 0, 0),
            )
            .rotate(
                PI / 6,
                axis=RIGHT,
                about_point=axes_3d.c2p(0, 0, 0),
            )
        )
        cube.move_to(axes_3d.c2p(0.5, 0.5, 0.5))
        self.play(FadeIn(cube))
        self.wait(0.3)

        arrow_a = Arrow3D(
            start=axes_3d.c2p(0, 0, 0),
            end=axes_3d.c2p(-0.0302, 0.7046, 0.0516),
            thickness=0.01,  # 轴的粗细
            color=c3,  # 颜色可选
        )

        arrow_b = Arrow3D(
            start=axes_3d.c2p(0, 0, 0),
            end=axes_3d.c2p(0.6100, 0, 0.3576),
            thickness=0.01,  # 轴的粗细
            color=c4,
        )

        def plane_func(u, v):
            a = np.array([-0.0302, 0.7046, 0.0516])  # arrow_a
            b = np.array([0.6100, 0, 0.3576])  # arrow_b
            p = u * a + v * b
            return axes_3d.c2p(*p)

        plane = Surface(
            lambda u, v: plane_func(u, v),
            u_range=[-3, 3],
            v_range=[-3, 3],
            fill_opacity=0.1,
            resolution=(16, 16),
        )

        self.play(FadeIn(plane), FadeIn(arrow_a), FadeIn(arrow_b))
        self.wait()

        def rotation_matrix_3d(k: float):
            P = np.array(
                [
                    [-0.50, -0.03 + 0.61j, -0.03 - 0.61j],
                    [-0.08, 0.70, 0.70],
                    [0.86, 0.05 + 0.36j, 0.05 - 0.36j],
                ]
            )
            D = np.array(
                [
                    [1, 0, 0],
                    [0, (-0.6469 + 0.7626j) ** k, 0],
                    [0, 0, (-0.6469 - 0.7626j) ** k],
                ]
            )
            T = P @ D @ np.linalg.inv(P)
            return T

        def axes_basis_matrix(axes: ThreeDAxes):
            # 取当前三轴的世界坐标方向（包含尺度），组列向量为基底矩阵 B
            x0, x1 = axes.x_axis.get_start_and_end()
            y0, y1 = axes.y_axis.get_start_and_end()
            z0, z1 = axes.z_axis.get_start_and_end()
            X = x1 - x0
            Y = y1 - y0
            Z = z1 - z0
            return np.column_stack([X, Y, Z])  # 3x3

        def local_rotation_matrix(axes: ThreeDAxes, lmbd: float):
            T_local = rotation_matrix_3d(lmbd)  # 在 axes_3d 的局部坐标
            B = axes_basis_matrix(axes)  # 局部->世界 的基底
            Binv = np.linalg.inv(B)
            return B @ T_local @ Binv  # 共轭到世界坐标

        def local_z_axis_dir():
            a, b = axes_3d.z_axis.get_start_and_end()
            return normalize(b - a)

        k_tracker = ValueTracker(0)

        cube_trans = cube.copy().set_color(c2)
        cube_trans.apply_matrix(
            local_rotation_matrix(axes_3d, 1), about_point=axes_3d.c2p(0, 0, 0)
        )

        cube_mid = always_redraw(
            lambda: cube.copy()
            .set_color(WHITE)
            .set_fill(c2, opacity=0)
            .apply_matrix(
                local_rotation_matrix(axes_3d, k_tracker.get_value()),
                about_point=axes_3d.c2p(0, 0, 0),
            )
        )
        self.play(FadeIn(cube_trans), FadeIn(cube_mid))

        grp = VGroup(axes_3d, cube, cube_mid, cube_trans, arrow_a, arrow_b, plane)
        grp.add_updater(
            lambda m, dt: m.rotate(
                0.6 * dt,
                axis=local_z_axis_dir(),
                about_point=axes_3d.c2p(0, 0, 0),
            )
        )

        self.add(grp)
        self.play(k_tracker.animate.set_value(1), run_time=8, rate_func=linear)
        self.wait(2)
