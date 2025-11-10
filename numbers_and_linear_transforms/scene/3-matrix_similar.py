from manim import *
from utils import *


class SimilarScene(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo)
        txt = Text("矩阵相似", font_size=38, color=WHITE)
        txt.to_edge(UL, buff=0.2)
        self.play(
            Write(txt),
        )

        transform_title = Text("变换矩阵", font_size=28)
        base_title = Text("基矩阵", font_size=28)
        titles = VGroup(transform_title, base_title).arrange(RIGHT, buff=1.5)
        titles.to_edge(UR, buff=0.5).shift(LEFT * 1)

        axes = NumberPlane(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=6,
            y_length=6,
        ).add_coordinates()
        axes.to_edge(LEFT, buff=1.2)  # 关键：整体靠左

        self.play(FadeIn(axes))

        mat_tex = MathTex(
            r"\mathbf{T} = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )

        mat_tex.next_to(transform_title, DOWN, buff=0.8)
        self.play(FadeIn(transform_title))
        self.play(FadeIn(mat_tex, shift=UP))
        self.wait(0.3)

        square = Square(side_length=1, color=c1, fill_opacity=0.4)
        square.move_to(axes.c2p(0, 0, 0))  # 依然用 axes 坐标

        self.play(FadeIn(square))

        mat = np.array([[1, 0.5], [0.5, 1]])
        P = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        Pinv = P.T

        square_copy = square.copy()
        square_copy.set_color(c2)

        axes_copy = axes.copy()
        self.bring_to_back(axes_copy)

        self.play(
            ApplyMatrix(mat, square_copy, axes.c2p(0, 0)),
            run_time=2,
        )

        self.play(
            axes.animate.set_color(GRAY).set_opacity(0.4),
            ApplyMatrix(P, axes_copy, axes.c2p(0, 0)),
            run_time=2,
        )

        self.play(
            ApplyMatrix(Pinv, axes_copy, axes.c2p(0, 0)),
            ApplyMatrix(Pinv, axes, axes.c2p(0, 0)),
            ApplyMatrix(Pinv, square, axes.c2p(0, 0)),
            ApplyMatrix(Pinv, square_copy, axes.c2p(0, 0)),
            run_time=2,
        )

        self.play(FadeOut(square_copy))

        square_copy = square.copy()
        square_copy.set_color(c2)

        self.play(ApplyMatrix(P.T @ mat @ P, square_copy, axes.c2p(0, 0)), run_time=2)

        base_mat = MathTex(
            r"\mathbf{B} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )
        base_mat.next_to(base_title, DOWN, buff=0.8)

        self.play(FadeIn(base_title))
        self.play(FadeIn(base_mat, shift=UP))
        self.wait(1)

        new_base_mat = MathTex(
            r"\mathbf{B'} = \begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}  \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )
        new_base_mat.next_to(base_mat, DOWN, buff=0.8)

        self.play(FadeIn(new_base_mat, shift=UP))

        new_mat_tex = MathTex(
            r"\mathbf{T'} = \begin{bmatrix} 1.5 & 0 \\ 0 & 0.5 \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )
        new_mat_tex.next_to(mat_tex, DOWN, buff=0.8)

        self.play(FadeIn(new_mat_tex, shift=UP))

        type_txt = Text("缩放变换", font_size=28, color=WHITE).next_to(
            titles, DOWN, buff=4
        )

        self.play(FadeIn(type_txt))

        underline = Line(
            type_txt.get_left() + DOWN * 0.3,
            type_txt.get_right() + DOWN * 0.3,
            color=WHITE,
        )

        self.play(Create(underline))
        self.wait(0.3)
        self.play(FadeOut(underline))

        # clear all
        self.play(
            FadeOut(axes),
            FadeOut(axes_copy),
            FadeOut(square),
            FadeOut(square_copy),
            FadeOut(mat_tex),
            FadeOut(transform_title),
            FadeOut(base_title),
            FadeOut(base_mat),
            FadeOut(new_base_mat),
            FadeOut(new_mat_tex),
            FadeOut(type_txt),
        )

        vec1 = MathTex(r"\boldsymbol{x}", color=c1)
        txt1 = Text("向量", font_size=32, color=c1)

        vgroup1 = VGroup(txt1, vec1)
        vgroup1.arrange(DOWN, buff=0.2)

        txt_and = Text("和", font_size=32, color=WHITE)

        vec2 = MathTex(r"\boldsymbol{x}_{\mathbf{B}}", color=c2)
        txt2 = Text("基表示", font_size=32, color=c2)

        vgroup2 = VGroup(txt2, vec2)
        vgroup2.arrange(DOWN, buff=0.2)

        vgroup_all = VGroup(vgroup1, txt_and, vgroup2)
        vgroup_all.arrange(RIGHT, buff=0.5).to_edge(UR, buff=2).shift(UP)

        self.play(Write(vgroup_all))

        axes = NumberPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=5,
            y_length=5,
        ).add_coordinates()
        axes.to_edge(LEFT, buff=1.2)  # 关键：整体靠左

        v = Arrow(
            start=axes.c2p(0, 0), end=axes.c2p(1, 1), buff=0, color=c1  # 不留空隙
        )
        self.play(Create(v))

        self.play(FadeIn(axes))

        base1 = MathTex(
            r"\mathbf{B}_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )

        vec1 = MathTex(
            r"\boldsymbol{x}_{\mathbf{B}_1} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}",
            color=c2,
        )

        vgroup3 = VGroup(base1, vec1)
        vgroup3.arrange(RIGHT, buff=0.5)

        vgroup3.next_to(vgroup_all, DOWN, buff=0.5)

        self.play(FadeIn(vgroup3, shift=UP))

        axes_copy = axes.copy()

        self.bring_to_back(axes_copy)

        self.play(
            ApplyMatrix(
                np.array([[1, -1], [1, 1]]) / np.sqrt(2), axes_copy, axes.c2p(0, 0)
            ),
            axes.animate.set_color(GRAY).set_opacity(0.4),
        )

        base2 = MathTex(
            r"\mathbf{B}_2 = \begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}  \end{bmatrix}",
            font_size=32,
            color=WHITE,
        )

        vec2 = MathTex(
            r"\boldsymbol{x}_{\mathbf{B}_2} = \begin{bmatrix} \sqrt{2} \\ 0 \end{bmatrix}",
            color=c2,
        )

        vgroup4 = VGroup(base2, vec2)
        vgroup4.arrange(RIGHT, buff=0.5)

        vgroup4.next_to(vgroup3, DOWN, buff=0.5)

        self.play(FadeIn(vgroup4, shift=UP))

        tex_eq = MathTex(
            r"\mathbf{B}_1 \boldsymbol{x}_{\mathbf{B}_1} = \mathbf{B}_2 \boldsymbol{x}_{\mathbf{B}_2}",
            font_size=32,
            color=WHITE,
        )
        tex_eq.next_to(vgroup4, DOWN, buff=0.5)
        self.play(Write(tex_eq))
        self.wait(1)

        self.play(
            FadeOut(vgroup3),
            FadeOut(vgroup4),
            FadeOut(axes),
            FadeOut(axes_copy),
            FadeOut(v),
            FadeOut(vgroup_all),
            FadeOut(tex_eq),
        )
        txt1 = Text("向量：", font_size=32, color=WHITE)
        eq1 = MathTex(r"\boldsymbol{x} = \mathbf{B} \boldsymbol{x}_{\mathbf{B}}")

        group1 = VGroup(txt1, eq1).arrange(RIGHT, buff=0.2).to_edge(UL, buff=1)

        self.play(Write(group1))

        txt2 = Text("线性变换：", font_size=32, color=WHITE)
        eq2 = MathTex(r"\mathbf{T} = f\left( \mathbf{T}_{\mathbf{B}} \right) ?")

        group2 = (
            VGroup(txt2, eq2).arrange(RIGHT, buff=0.2).next_to(group1, RIGHT, buff=0.5)
        )

        self.play(FadeIn(group2))

        axes1 = NumberPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=3,
            y_length=3,
        ).add_coordinates()

        axes2 = NumberPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=3,
            y_length=3,
        ).add_coordinates()

        axes2.apply_matrix(
            np.array([[1, -1], [1, 1]]) / np.sqrt(2), about_point=axes2.c2p(0, 0)
        )

        axes_group = VGroup(axes1, axes2).arrange(RIGHT, buff=1.5)

        vec1 = Arrow(start=axes1.c2p(0, 0), end=axes1.c2p(1, 1), buff=0, color=c1)
        vec2 = Arrow(
            start=axes2.c2p(0, 0), end=axes2.c2p(np.sqrt(2), 0), buff=0, color=c1
        )

        lab1 = MathTex(r"\mathbf{B}_1", font_size=32).next_to(axes1, UP, buff=0.2)
        lab2 = MathTex(r"\mathbf{B}_2", font_size=32).next_to(axes2, UP, buff=0.2)

        self.play(FadeIn(axes_group), Write(lab1), Write(lab2))
        self.play(Create(vec1), Create(vec2))

        f1 = MathTex(
            r"\boldsymbol{y}_{\mathbf{B}_1} = \mathbf{T}_{\mathbf{B}_1} \boldsymbol{x}_{\mathbf{B}_1}",
            font_size=32,
            color=WHITE,
        ).next_to(axes1, DOWN, buff=0.2)

        f2 = MathTex(
            r"\boldsymbol{y}_{\mathbf{B}_2} = \mathbf{T}_{\mathbf{B}_2} \boldsymbol{x}_{\mathbf{B}_2}",
            font_size=32,
            color=WHITE,
        ).next_to(axes2, DOWN, buff=0.2)

        vec1_copy = vec1.copy()
        vec2_copy = vec2.copy()

        theta = np.deg2rad(30)  # 转弧度
        R30 = (
            np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            * 1.5
        )

        v1 = np.array([1, 1])
        v1_ = R30 @ v1
        vec1_rotated = Arrow(
            start=axes1.c2p(0, 0),
            end=axes1.c2p(v1_[0], v1_[1]),
            buff=0,
            color=c2,  # 不留空隙
        )
        v2 = np.array([np.sqrt(2), 0])
        v2_ = R30 @ v2
        vec2_rotated = Arrow(
            start=axes2.c2p(0, 0),
            end=axes2.c2p(v2_[0], v2_[1]),
            buff=0,
            color=c2,  # 不留空隙
        )

        self.play(
            Transform(vec1_copy, vec1_rotated), Transform(vec2_copy, vec2_rotated)
        )
        self.play(Write(f1), Write(f2))

        self.play(
            FadeOut(axes_group),
            FadeOut(lab1),
            FadeOut(lab2),
            FadeOut(vec1_copy),
            FadeOut(vec2_copy),
            FadeOut(vec1),
            FadeOut(vec2),
        )

        self.play(f1.animate.shift(UP * 3.5), f2.animate.shift(UP * 3.5))

        f1_ = MathTex(
            r"\mathbf{B}_1 \boldsymbol{y}_{\mathbf{B}_1} = \mathbf{B}_1 \mathbf{T}_{\mathbf{B}_1} \boldsymbol{x}_{\mathbf{B}_1}",
            font_size=32,
            color=WHITE,
        )
        f1_.move_to(f1)

        f2_ = MathTex(
            r"\mathbf{B}_2 \boldsymbol{y}_{\mathbf{B}_2} = \mathbf{B}_2 \mathbf{T}_{\mathbf{B}_2} \boldsymbol{x}_{\mathbf{B}_2}",
            font_size=32,
            color=WHITE,
        )
        f2_.move_to(f2)

        self.play(Transform(f1, f1_), Transform(f2, f2_))

        eq_syb = MathTex(
            r"\mathbf{B}_1 \mathbf{T}_{\mathbf{B}_1} \boldsymbol{x}_{\mathbf{B}_1}=\mathbf{B}_2 \mathbf{T}_{\mathbf{B}_2} \boldsymbol{x}_{\mathbf{B}_2}",
            font_size=32,
        )
        eq_syb.move_to((f1_.get_center() + f2_.get_center()) / 2).shift(DOWN * 1)
        self.play(Write(eq_syb))

        eq_syb2 = MathTex(
            r"\mathbf{B}_1 \mathbf{T}_{\mathbf{B}_1} \mathbf{B}_1^{-1} \boldsymbol{x} =\mathbf{B}_2 \mathbf{T}_{\mathbf{B}_2} \mathbf{B}_2^{-1} \boldsymbol{x}",
            font_size=32,
        ).move_to(eq_syb)

        self.play(Transform(eq_syb, eq_syb2))

        eq_syb3 = MathTex(
            r"\mathbf{B}_1 \mathbf{T}_{\mathbf{B}_1} \mathbf{B}_1^{-1} =\mathbf{B}_2 \mathbf{T}_{\mathbf{B}_2} \mathbf{B}_2^{-1}",
            font_size=32,
        ).move_to(eq_syb)

        self.play(Transform(eq_syb, eq_syb3))

        eq_syb_copy = eq_syb.copy()

        eq_syb4 = MathTex(
            r"\mathbf{T}_{\mathbf{B}_1} =\mathbf{B}_1^{-1} \mathbf{B}_2 \mathbf{T}_{\mathbf{B}_2} \mathbf{B}_2^{-1}  \mathbf{B}_1",
            font_size=32,
        ).shift(DOWN * 0.5)

        self.play(Transform(eq_syb_copy, eq_syb4))

        eq_syb5 = MathTex(
            r"\mathbf{T}_{\mathbf{B}_1} = (\mathbf{B}_1^{-1} \mathbf{B}_2) \mathbf{T}_{\mathbf{B}_2} (\mathbf{B}_1^{-1} \mathbf{B}_2)^{-1}",
            font_size=32,
        ).shift(DOWN * 0.5)

        self.play(Transform(eq_syb_copy, eq_syb5))

        tex_group = (
            VGroup(
                MathTex(
                    r"\mathbf{P} = \mathbf{B}_1^{-1} \mathbf{B}_2 \rightarrow \mathbf{T}_{\mathbf{B}_1} = \mathbf{P} \mathbf{T}_{\mathbf{B}_2} \mathbf{P}^{-1}"
                ).set_font_size(32),
            )
            .arrange(RIGHT, buff=0.2)
            .next_to(eq_syb5, DOWN, buff=0.5)
        )

        self.play(Write(tex_group))
        self.wait(1)

        eq2_new = (
            MathTex(r"\mathbf{T} = \mathbf{B} \mathbf{T}_{\mathbf{B}} \mathbf{B}^{-1}")
            .shift(DOWN * 1)
            .move_to(eq2)
            .align_to(eq2, LEFT)
        )

        self.play(Transform(eq2, eq2_new))

        box = SurroundingRectangle(eq2_new, color=c1, buff=0.1)
        self.play(Create(box))
        self.wait(2)
