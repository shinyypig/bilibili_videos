from manim import *
from utils import *


class IntroScene(Scene):
    def apply_transformation(self, mat, mat_tex, transform_name, obj, color=c2):
        mat_tex.to_edge(LEFT, buff=0.5)
        self.play(FadeIn(mat_tex, shift=UP))
        self.wait(0.3)
        obj_trans = obj.copy()
        obj_trans.set_color(color)
        self.play(ApplyMatrix(mat, obj_trans), run_time=2)
        if transform_name:
            self.play(FadeIn(transform_name, shift=UP, run_time=1))
        self.wait(0.5)
        if transform_name:
            self.play(
                FadeOut(obj_trans),
                FadeOut(mat_tex),
                transform_name.animate.set_color(GRAY),
            )
        else:
            self.play(
                FadeOut(obj_trans),
                FadeOut(mat_tex),
            )

    def construct(self):
        logo = Logo()
        self.add(logo)

        # 线性变换
        txt = Text("简介", font_size=38, color=WHITE)
        txt.to_edge(UL, buff=0.2)
        self.play(
            Write(txt),
            run_time=2,
            subcaption="我们知道nxn的矩阵可以表示一个线性变换",
            subcaption_duration=7,
        )
        self.wait(5)

        mtxt = MathTex(
            r"\mathbf{T} \in \mathbb{R}^{n \times n}: \mathbb{R}^{n} \to \mathbb{R}^{n}",
            font_size=32,
            color=WHITE,
        )
        mtxt.to_edge(UP, buff=1.5)

        self.play(
            Write(mtxt),
            run_time=2,
            subcaption="而该变换可以将n维线性空间中的元素\n线性映射到该空间中的另一个元素",
            subcaption_duration=6,
        )
        self.wait(8)
        self.play(FadeOut(mtxt))

        scale_txt = Text("1. 缩放变换", font_size=32, color=WHITE)
        proj_txt = Text("2. 投影变换", font_size=32, color=WHITE)
        reflect_txt = Text("3. 反射变换", font_size=32, color=WHITE)
        shear_txt = Text("4. 剪切变换", font_size=32, color=WHITE)
        rotate_txt = Text("5. 旋转变换", font_size=32, color=WHITE)

        vgroup = VGroup(
            scale_txt, proj_txt, reflect_txt, shear_txt, rotate_txt
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        vgroup.to_edge(UR, buff=0.5)

        axes = NumberPlane(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=6,
            y_length=6,
        ).add_coordinates()
        self.add(axes)

        square = Square(side_length=1, color=c1, fill_opacity=0.4)
        square.move_to([0.5, 0.5, 0])  # 放在第一象限
        self.play(FadeIn(square))

        mat_list = [
            np.array([[2, 0], [0, 1]]),
            np.array([[0, 0], [0, 1]]),
            np.array([[-1, 0], [0, 1]]),
            np.array([[1, 1], [0, 1]]),
            np.array(
                [
                    [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4)],
                ]
            ),
        ]

        mat_tex_list = [
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} \cos\frac{\pi}{4} & -\sin\frac{\pi}{4} \\ \sin\frac{\pi}{4} & \cos\frac{\pi}{4} \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
        ]

        for i in range(len(mat_list)):
            self.apply_transformation(
                mat_list[i],
                mat_tex_list[i],
                vgroup[i],
                square,
                c2,
            )

        mat_list = [
            np.array([[2, 1], [1, 2]]),
            np.array([[1, 3], [2, 1]]) / 2,
        ]

        mat_tex_list = [
            MathTex(
                r"\mathbf{T} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
            MathTex(
                r"\mathbf{T} = \frac{1}{2} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}",
                font_size=32,
                color=WHITE,
            ),
        ]

        for i in range(len(mat_list)):
            self.apply_transformation(
                mat_list[i],
                mat_tex_list[i],
                None,
                square,
                c2,
            )

        # clear page
        self.play(FadeOut(vgroup), FadeOut(square), FadeOut(axes))

        question1 = Text(
            "1. 给定一个任意的矩阵，该如何判断它包含了哪些线性变换？", font_size=28
        )

        question2 = Text("2. 线性变换共有多少种？", font_size=28)

        question1.to_edge(UP, buff=2)
        question2.next_to(question1, DOWN, buff=1)
        question2.align_to(question1, LEFT)

        self.play(
            FadeIn(question1, shift=UP),
        )

        self.play(
            FadeIn(question2, shift=UP),
        )

        new_transform = Text("6. ？？？", font_size=32, color=c2)
        vgroup.set_color(WHITE)
        vgroup.add(new_transform).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        self.play(
            vgroup.animate.arrange(RIGHT, aligned_edge=UP, buff=0.5)
            .next_to(question2, DOWN, buff=0.5)
            .scale(0.7)
            .align_to(question1, LEFT)
        )
