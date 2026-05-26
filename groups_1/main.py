# import necessary libraries
from manim import *
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from utils import *


def reflect_matrix(theta):
    c2t = np.cos(2 * theta)
    s2t = np.sin(2 * theta)
    return np.array(
        [
            [c2t, s2t, 0.0],
            [s2t, -c2t, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotate_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def ArrowFish(size=0.55):
    """
    一个不对称的“箭头小鱼”，反射后非常明显：
    - 右边尖嘴(arrow tip)
    - 左边三角尾巴
    - 上方小背鳍
    """
    # 主体：不对称多边形（右尖、左钝）
    body_pts = np.array(
        [
            [-1.10, -0.35, 0],
            [0.55, -0.35, 0],
            [1.15, 0.00, 0],  # 嘴尖
            [0.55, 0.35, 0],
            [-1.10, 0.35, 0],
            [-0.65, 0.00, 0],  # 左侧内凹（更不对称）
        ]
    )
    body = Polygon(*body_pts, stroke_width=3)

    # 尾巴：左侧三角形
    tail = Polygon(
        [-1.10, 0.00, 0], [-1.55, 0.35, 0], [-1.55, -0.35, 0], stroke_width=3
    )

    # 背鳍：上方小三角（偏右）
    fin = Polygon([0.05, 0.35, 0], [0.35, 0.70, 0], [0.55, 0.35, 0], stroke_width=3)

    # 眼睛：一个小圆点（偏右上）
    eye = Dot(point=[0.65, 0.12, 0], radius=0.06)

    fish = VGroup(body, tail, fin, eye)
    fish.set_fill(opacity=0.8)
    fish.set_stroke(color=WHITE)
    body.set_fill(color=TEAL)
    tail.set_fill(color=GREEN)
    fin.set_fill(color=ORANGE)
    eye.set_color(BLACK)

    fish.scale(size)
    return fish


def SquareWithDots(side_length=2, dot_radius=0.05):
    square = Square(side_length=side_length, stroke_width=3)
    square.set_fill(BLUE_E, opacity=0.5)

    # 在四个顶点放置小圆点
    corners = [
        square.get_corner(UL),
        square.get_corner(UR),
        square.get_corner(DR),
        square.get_corner(DL),
    ]
    colors = [c2, c3, c4, c5]
    dots = VGroup(
        *[
            Dot(point=corner, radius=dot_radius, color=color)
            for corner, color in zip(corners, colors)
        ]
    )

    square_with_dots = VGroup(square, dots)
    return square_with_dots


# Scene 1: Welcome Scene
class WelcomeScene(Scene):
    def construct(self):
        ## Show the welcome title and logo
        title = Text("两次反射等于旋转", font_size=80, color=WHITE)
        subtitle = Text("—群论视角", font_size=48, color=WHITE)
        title.add(subtitle.next_to(title, DOWN, buff=0.5, aligned_edge=RIGHT))
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


# Scene 2: Prove in Geometry
class ProveInGeometry(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("几何证明", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.play(Write(title))

        plane = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=5,
            y_length=5,
            axis_config={
                "include_tip": True,
                "tip_width": 0.2,
                "stroke_width": 1.5,
                "include_ticks": False,  # 不显示一格一格的刻度
                "include_numbers": False,  # 不显示数字
            },
        )
        plane.to_edge(LEFT, buff=1)

        self.play(Create(plane))

        theta1 = 60 * DEGREES

        line1 = DashedLine(
            start=plane.c2p(-2, -2 * np.tan(theta1)),
            end=plane.c2p(2, 2 * np.tan(theta1)),
            color=c1,
            dash_length=0.2,
        )
        self.play(Create(line1))

        # draw an arc at origin to indicate the angle between x-axis and line1
        arc_center = plane.c2p(0, 0)
        arc_radius = 0.3
        angle_arc = Arc(
            arc_center=arc_center,
            radius=arc_radius,
            start_angle=0,
            angle=theta1,
            color=c1,
            stroke_width=4,
        )
        angle_label = MathTex(r"\theta", font_size=36)
        mid_angle = theta1 / 2
        label_pos = plane.c2p(
            arc_radius * np.cos(mid_angle) * 3, arc_radius * np.sin(mid_angle) * 3
        )
        angle_label.move_to(label_pos)

        self.play(Create(angle_arc), Write(angle_label), run_time=1)

        # 将角度标志（弧线与标签）变暗为灰白色
        self.play(
            angle_arc.animate.set_stroke(color=GREY_B, opacity=0.4),
            angle_label.animate.set_color(GREY_B).set_opacity(0.6),
            run_time=0.8,
        )

        alpha = 80 * DEGREES
        arrow_len = 2.5
        arrow_alpha = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(arrow_len * np.cos(alpha), arrow_len * np.sin(alpha)),
            buff=0,
            color=c2,
            stroke_width=4,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.12,  # 让箭头尖更小
        )
        self.play(GrowArrow(arrow_alpha), run_time=1.2)

        # angle marker for alpha (between x-axis and arrow_alpha), placed underneath
        arc_radius_alpha = 0.5
        alpha_arc = Arc(
            arc_center=plane.c2p(0, 0),
            radius=arc_radius_alpha,
            start_angle=0,
            angle=alpha,
            color=c2,
            stroke_width=4,
        ).set_z_index(-2)

        alpha_label = MathTex(r"\alpha", font_size=34).set_z_index(-2)
        mid_alpha = alpha / 2
        alpha_label.move_to(
            plane.c2p(
                arc_radius_alpha * 3 * np.cos(mid_alpha),
                arc_radius_alpha * 3 * np.sin(mid_alpha),
            )
        )

        self.play(Create(alpha_arc), Write(alpha_label), run_time=1)

        # dim the angle marker
        self.play(
            alpha_arc.animate.set_stroke(color=GREY_B, opacity=0.35),
            alpha_label.animate.set_color(GREY_B).set_opacity(0.55),
            run_time=0.8,
        )

        # reflected arrow across line1 (line through origin making angle theta1 with x-axis)
        ref_alpha = 2 * theta1 - alpha
        arrow_reflected = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(arrow_len * np.cos(ref_alpha), arrow_len * np.sin(ref_alpha)),
            buff=0,
            color=c3 if "c3" in globals() else WHITE,
            stroke_width=4,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.12,
        )

        self.play(GrowArrow(arrow_reflected), run_time=1.2)

        # angle marker for reflected angle (between x-axis and arrow_reflected)
        arc_radius_ref = 0.65
        ref_arc = Arc(
            arc_center=plane.c2p(0, 0),
            radius=arc_radius_ref,
            start_angle=0,
            angle=ref_alpha,
            color=arrow_reflected.get_color(),
            stroke_width=4,
        ).set_z_index(-2)

        ref_label = MathTex(r"\beta", font_size=34).set_z_index(-2)
        mid_ref = ref_alpha / 2
        ref_label.move_to(
            plane.c2p(
                arc_radius_ref * 3 * np.cos(mid_ref),
                arc_radius_ref * 3 * np.sin(mid_ref),
            )
        )

        self.play(Create(ref_arc), Write(ref_label), run_time=1)

        # dim the angle marker
        self.play(
            ref_arc.animate.set_stroke(color=GREY_B, opacity=0.35),
            ref_label.animate.set_color(GREY_B).set_opacity(0.55),
            run_time=0.8,
        )

        # show that beta = 2theta - alpha
        eq1 = MathTex(r"\alpha - \theta = \theta - \beta", font_size=48)

        eq1.next_to(title, DOWN + RIGHT * 1.5, buff=0.5)
        self.play(Write(eq1), run_time=2)

        eq2 = MathTex(r"\beta = 2\theta - \alpha", font_size=48)
        eq2.next_to(eq1, DOWN, buff=0.5)
        self.play(TransformFromCopy(eq1, eq2), run_time=2)

        eq3 = MathTex(r"F_{\theta} (\alpha) = 2\theta - \alpha", font_size=48)
        eq3.next_to(eq2, DOWN, buff=0.5)
        self.play(Write(eq3), run_time=2)

        eq4 = MathTex(
            r"F_{\theta_2} \circ F_{\theta_1} (\alpha) = \alpha + 2(\theta_2 - \theta_1)",
            font_size=48,
        )
        eq4.next_to(eq3, DOWN, buff=0.5)
        self.play(Write(eq4), run_time=4)

        txt1 = Text("两次反射等于旋转", font_size=36)
        txt1.next_to(eq4, DOWN, buff=0.5)
        self.play(Write(txt1), run_time=2)


# Scene 3: an example scene
class Example(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("几何证明", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.play(Write(title))

        plane1 = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=5,
            y_length=5,
            axis_config={
                "include_tip": True,
                "tip_width": 0.2,
                "stroke_width": 1.5,
                "include_ticks": False,  # 不显示一格一格的刻度
                "include_numbers": False,  # 不显示数字
            },
        )
        plane1.to_edge(LEFT, buff=1)

        plane2 = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=5,
            y_length=5,
            axis_config={
                "include_tip": True,
                "tip_width": 0.2,
                "stroke_width": 1.5,
                "include_ticks": False,  # 不显示一格一格的刻度
                "include_numbers": False,  # 不显示数字
            },
        )
        plane2.to_edge(RIGHT, buff=1)

        self.play(Create(plane1), Create(plane2))

        theta1 = 30 * DEGREES
        theta2 = 60 * DEGREES

        line1 = DashedLine(
            start=plane1.c2p(-4 * np.cos(theta1), -4 * np.sin(theta1)),
            end=plane1.c2p(4 * np.cos(theta1), 4 * np.sin(theta1)),
            color=c1,
            dash_length=0.2,
        )

        line2 = DashedLine(
            start=plane1.c2p(-4 * np.cos(theta2), -4 * np.sin(theta2)),
            end=plane1.c2p(4 * np.cos(theta2), 4 * np.sin(theta2)),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line1), Create(line2), run_time=0.2)

        line1 = DashedLine(
            start=plane2.c2p(-4 * np.cos(theta1), -4 * np.sin(theta1)),
            end=plane2.c2p(4 * np.cos(theta1), 4 * np.sin(theta1)),
            color=c1,
            dash_length=0.2,
        )

        line2 = DashedLine(
            start=plane2.c2p(-4 * np.cos(theta2), -4 * np.sin(theta2)),
            end=plane2.c2p(4 * np.cos(theta2), 4 * np.sin(theta2)),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line1), Create(line2), run_time=0.2)

        # 造“几何猫”（这里用 ArrowFish）
        cat = ArrowFish(size=0.55)
        cat.move_to(plane1.c2p(2, 1))
        self.play(FadeIn(cat), run_time=1)

        origin1 = plane1.c2p(0, 0)

        theta1 = 30 * DEGREES
        theta2 = 60 * DEGREES

        R1 = reflect_matrix(theta1)
        R2 = reflect_matrix(theta2)

        ghost1 = cat.copy()
        self.add(ghost1)
        self.play(ApplyMatrix(R1, ghost1, about_point=origin1), run_time=1.2)
        self.play(
            ApplyMatrix(R2, ghost1, about_point=origin1),
            run_time=1.2,
        )
        self.play(
            ghost1.animate.set_opacity(1),
            run_time=0.1,
        )

        # 造第二只“几何猫”（这里用 ArrowFish）
        cat2 = ArrowFish(size=0.55)
        cat2.move_to(plane2.c2p(2, 1))
        self.play(FadeIn(cat2), run_time=1)

        origin2 = plane2.c2p(0, 0)
        # 角度跟踪器：从 0 连续变到目标角
        phi = ValueTracker(0.0)
        # 每一帧都“重画”一只 = base.copy() + apply_matrix(随phi变化)
        ghost2 = always_redraw(
            lambda: cat2.copy().apply_matrix(
                rotate_matrix(phi.get_value()), about_point=origin2
            )
        )

        self.add(ghost2)
        # 连续变角：就是在变 tracker
        self.play(
            phi.animate.set_value(2 * (theta2 - theta1)), run_time=2, rate_func=linear
        )


class IntroToGroups(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("群论入门", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.play(Write(title))

        # 给出群的定义（放在页面左侧）
        group_def = Text(
            "群 (G, o) 是一个集合 G 和一个二元运算 o 的组合，满足：",
            font_size=32,
        )
        group_def.to_edge(LEFT, buff=0.5).to_edge(UP, buff=1.5)
        self.play(Write(group_def), run_time=3)
        # 群的四条公理
        axioms = VGroup(
            Text(
                "1. 封闭性：任意 a, b 在 G 中，a o b 仍在 G 中",
                font_size=32,
            ),
            Text(
                "2. 结合性：(a o b) o c = a o (b o c)",
                font_size=32,
            ),
            Text(
                "3. 单位元：存在 e，使得 e o a = a o e = a",
                font_size=32,
            ),
            Text(
                "4. 逆元：每个 a 都有 b，使得 a o b = b o a = e",
                font_size=32,
            ),
        )
        for i, axiom in enumerate(axioms):
            axiom.next_to(group_def, DOWN, buff=0.5 + i * 0.6, aligned_edge=LEFT)
            self.play(Write(axiom), run_time=2)
        self.wait(2)

        # 消去以上内容
        self.play(
            FadeOut(group_def),
            FadeOut(axioms),
            shift=UP,
            run_time=2,
        )

        # 展示D4群
        square = SquareWithDots(side_length=2, dot_radius=0.06)
        square.to_edge(LEFT, buff=1)
        self.play(Create(square), run_time=2)

        square_move = square

        txt1 = Text("反射", font_size=36)
        txt1.next_to(title, DOWN + RIGHT * 3, buff=0.5)
        self.play(Write(txt1), run_time=2)

        line_reflect = DashedLine(
            start=square_move.get_top(),
            end=square_move.get_bottom(),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line_reflect), run_time=0.5)

        square_reflected = square_move.copy()
        square_reflected.flip(axis=line_reflect.get_start() - line_reflect.get_end())

        F1 = MathTex(r"F_{0}", font_size=48)
        F1.next_to(txt1, RIGHT, buff=0.5)
        self.play(Transform(square_move, square_reflected), Write(F1), run_time=2)

        self.play(FadeOut(line_reflect), run_time=0.5)
        line_reflect = DashedLine(
            start=square_move.get_corner(UR),
            end=square_move.get_corner(DL),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line_reflect), run_time=0.5)
        square_reflected = square_move.copy()
        square_reflected.flip(axis=line_reflect.get_start() - line_reflect.get_end())

        F2 = MathTex(r"F_{\frac{\pi}{4}}", font_size=48)
        F2.next_to(F1, RIGHT, buff=0.5)
        self.play(Transform(square_move, square_reflected), Write(F2), run_time=2)

        # 第三个反射：水平对称轴
        self.play(FadeOut(line_reflect), run_time=0.5)
        line_reflect = DashedLine(
            start=square_move.get_left(),
            end=square_move.get_right(),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line_reflect), run_time=0.5)

        square_reflected = square_move.copy()
        square_reflected.flip(axis=line_reflect.get_start() - line_reflect.get_end())

        F3 = MathTex(r"F_{\frac{\pi}{2}}", font_size=48)
        F3.next_to(F2, RIGHT, buff=0.5)
        self.play(Transform(square_move, square_reflected), Write(F3), run_time=2)

        # 第四个反射：另一条对角线
        self.play(FadeOut(line_reflect), run_time=0.5)
        line_reflect = DashedLine(
            start=square_move.get_corner(UL),
            end=square_move.get_corner(DR),
            color=c2,
            dash_length=0.2,
        )
        self.play(Create(line_reflect), run_time=0.5)

        square_reflected = square_move.copy()
        square_reflected.flip(axis=line_reflect.get_start() - line_reflect.get_end())

        F4 = MathTex(r"F_{\frac{3\pi}{4}}", font_size=48)
        F4.next_to(F3, RIGHT, buff=0.5)
        self.play(Write(F4), Transform(square_move, square_reflected), run_time=2)

        self.play(FadeOut(line_reflect), run_time=0.5)
        self.wait(1)

        txt2 = Text("旋转", font_size=36)
        txt2.next_to(txt1, DOWN, buff=0.5)
        self.play(Write(txt2), run_time=1.2)

        R0 = MathTex(r"R_{0}", font_size=48)
        R0.next_to(txt2, RIGHT, buff=0.5)
        self.play(Write(R0), run_time=0.8)

        Itex = MathTex(r"I", font_size=48)
        Itex.next_to(txt2, RIGHT, buff=0.5)
        self.play(Transform(R0, Itex), run_time=0.8)
        self.wait(1)

        R90 = MathTex(r"R_{\frac{\pi}{2}}", font_size=48)
        R90.next_to(R0, RIGHT, buff=0.6)
        self.play(
            Rotate(square_move, angle=PI / 2, about_point=square_move.get_center()),
            Write(R90),
            run_time=1.2,
        )
        self.wait(1)

        R180 = MathTex(r"R_{\pi}", font_size=48)
        R180.next_to(R90, RIGHT, buff=0.6)
        self.play(
            Rotate(square_move, angle=PI, about_point=square_move.get_center()),
            Write(R180),
            run_time=2.4,
        )
        self.wait(1)

        R270 = MathTex(r"R_{\frac{3\pi}{2}}", font_size=48)
        R270.next_to(R180, RIGHT, buff=0.6)
        self.play(
            Rotate(square_move, angle=PI * 1.5, about_point=square_move.get_center()),
            Write(R270),
            run_time=3.6,
        )
        self.wait(1)

        Gtex = MathTex(
            r"G = \{ I, R_{\frac{\pi}{2}}, R_{\pi}, R_{\frac{3\pi}{2}},F_{0}, F_{\frac{\pi}{4}}, F_{\frac{\pi}{2}}, F_{\frac{3\pi}{4}} \}",
            font_size=36,
        )
        Gtex.next_to(txt2, DOWN, buff=0.7, aligned_edge=LEFT)
        self.play(Write(Gtex), run_time=3)
        self.wait(2)

        axioms = VGroup(
            Text(
                "1. 封闭性",
                font_size=32,
            ),
            Text(
                "2. 结合性",
                font_size=32,
            ),
            Text(
                "3. 单位元",
                font_size=32,
            ),
            Text(
                "4. 逆元",
                font_size=32,
            ),
        )
        for i, axiom in enumerate(axioms):
            axiom.next_to(Gtex, DOWN, buff=0.3 + i * 0.6, aligned_edge=LEFT)
            self.play(Write(axiom), run_time=2)
            self.wait(1)
        self.wait(2)


#
class ExampleofGroups(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("群论入门", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.add(title)

        # 展示Dn群
        txt1 = Text("二面体群：", font_size=32)
        tex1 = MathTex(
            r"\{D_n, \circ\}, \quad D_n = \{ R_k, F_k | k=0,1,...,n-1 \}", font_size=36
        )
        txt1.next_to(title, DOWN, buff=0.5)
        tex1.next_to(txt1, RIGHT, buff=0.2)
        self.play(Write(txt1), Write(tex1), run_time=3)

        # 整数加法群
        txt2 = Text("整数加法群：", font_size=32)
        tex2 = MathTex(r"\{\mathbb{Z}, +\}", font_size=36)
        txt2.next_to(txt1, DOWN, buff=0.5, aligned_edge=LEFT)
        tex2.next_to(txt2, RIGHT, buff=0.2)
        self.play(Write(txt2), Write(tex2), run_time=3)

        # 非零有理数乘法群
        txt3 = Text("非零有理数乘法群：", font_size=32)
        tex3 = MathTex(r"\{\mathbb{Q} \setminus \{0\}, \times\}", font_size=36)
        txt3.next_to(txt2, DOWN, buff=0.5, aligned_edge=LEFT)
        tex3.next_to(txt3, RIGHT, buff=0.2)
        self.play(Write(txt3), Write(tex3), run_time=3)

        # 一般线性群
        txt4 = Text("一般线性群：", font_size=32)
        tex4 = MathTex(r"\{GL(n, \mathbb{R}), \circ\}", font_size=36)
        txt4.next_to(txt3, DOWN, buff=0.5, aligned_edge=LEFT)
        tex4.next_to(txt4, RIGHT, buff=0.2)
        self.play(Write(txt4), Write(tex4), run_time=3)

        # 置换群
        txt5 = Text("置换群：", font_size=32)
        tex5 = MathTex(r"\{S_n, \circ\}", font_size=36)
        txt5.next_to(txt4, DOWN, buff=0.5, aligned_edge=LEFT)
        tex5.next_to(txt5, RIGHT, buff=0.2)
        self.play(Write(txt5), Write(tex5), run_time=3)

        # 性质
        axioms = VGroup(
            Text("封闭性", font_size=32),
            Text("结合性", font_size=32),
            Text("单位元", font_size=32),
            Text("逆元", font_size=32),
        )
        for i, axiom in enumerate(axioms):
            axiom.next_to(
                title,
                DOWN * 4 + RIGHT * 12 + DOWN * i * 1.2,
                buff=0.5,
                aligned_edge=LEFT,
            )
            self.play(Write(axiom), run_time=2)
            self.wait(1)


class Subgroup(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("子群", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.add(title)

        right_x = 2.25

        intro = VGroup(
            MathTex(r"H \leq G", font_size=56),
            Text(
                "如果 H 是 G 的一部分，并且 H 自己也是一个群，",
                font_size=32,
            ),
            Text(
                "那么 H 就叫做 G 的子群。",
                font_size=32,
            ),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        intro.to_edge(LEFT, buff=0.7).shift(UP * 0.8)

        self.play(Write(intro[0]), run_time=1.2)
        self.play(Write(intro[1]), Write(intro[2]), run_time=2.5)
        self.wait(1)

        shortcut = VGroup(
            Text(
                "判断子群，只需要检查：",
                font_size=32,
            ),
            MathTex(r"1.\ e \in H", font_size=32),
            MathTex(
                r"2.\ a,b \in H \Rightarrow a\circ b^{-1}\in H",
                font_size=32,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        shortcut.next_to(intro, DOWN, buff=0.6, aligned_edge=LEFT)

        self.play(Write(shortcut), run_time=3)
        self.wait(1.5)

        self.play(
            FadeOut(intro),
            FadeOut(shortcut),
            shift=UP,
            run_time=1.5,
        )

        # 用 D4 里的旋转来展示第一个子群。
        square = SquareWithDots(side_length=1.85, dot_radius=0.055)
        square.move_to(LEFT * 3.8 + DOWN * 0.35)
        square_label = MathTex(r"D_4", font_size=42)
        square_label.next_to(square, UP, buff=0.4)

        self.play(Create(square), Write(square_label), run_time=1.5)

        rot_title = Text("只取旋转", font_size=32)
        H_rot = MathTex(
            r"H=\{I,R_{\frac{\pi}{2}},R_{\pi},R_{\frac{3\pi}{2}}\}",
            font_size=34,
        )
        labels = VGroup(
            MathTex(r"I", font_size=34),
            MathTex(r"R_{\frac{\pi}{2}}", font_size=34),
            MathTex(r"R_{\pi}", font_size=34),
            MathTex(r"R_{\frac{3\pi}{2}}", font_size=34),
        ).arrange(RIGHT, buff=0.45)
        closed = MathTex(
            r"R_a\circ R_b=R_{a+b}\in H",
            font_size=36,
        )
        conclusion = MathTex(r"H \leq D_4", font_size=44, color=c2)
        rot_panel = VGroup(rot_title, H_rot, labels, closed, conclusion).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT
        )
        rot_panel.move_to(RIGHT * right_x + UP * 0.25)

        self.play(Write(rot_title), Write(H_rot), run_time=2)

        rotations = [
            (r"I", 0),
            (r"R_{\frac{\pi}{2}}", PI / 2),
            (r"R_{\pi}", PI),
            (r"R_{\frac{3\pi}{2}}", 3 * PI / 2),
        ]
        current_square = square
        for label, (_, angle) in zip(labels, rotations):
            if angle == 0:
                self.play(Write(label), run_time=0.7)
            else:
                self.play(
                    Rotate(
                        current_square,
                        angle=angle,
                        about_point=current_square.get_center(),
                    ),
                    Write(label),
                    run_time=1.2,
                )
            self.wait(0.2)
        self.play(Write(closed), run_time=1.8)

        self.play(Write(conclusion), run_time=1.2)
        self.wait(1.5)

        self.play(
            FadeOut(rot_title),
            FadeOut(H_rot),
            FadeOut(labels),
            FadeOut(closed),
            FadeOut(conclusion),
            run_time=1.2,
        )

        # 一个由单个反射生成的子群。
        reflect_title = Text("一次反射生成的子群", font_size=32)
        K_ref = MathTex(r"K=\{I,F_0\}", font_size=38)
        ref_rule = MathTex(r"F_0\circ F_0=I", font_size=38)
        K_conclusion = MathTex(r"K \leq D_4", font_size=44, color=c2)
        reflect_panel = VGroup(reflect_title, K_ref, ref_rule, K_conclusion).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT
        )
        reflect_panel.move_to(RIGHT * right_x + UP * 0.25)

        self.play(Write(reflect_title), Write(K_ref), run_time=1.8)

        reflect_axis = DashedLine(
            start=current_square.get_top(),
            end=current_square.get_bottom(),
            color=c2,
            dash_length=0.18,
        )
        self.play(Create(reflect_axis), run_time=0.7)

        reflected_square = current_square.copy()
        reflected_square.flip(axis=reflect_axis.get_start() - reflect_axis.get_end())
        self.play(Transform(current_square, reflected_square), run_time=1.0)

        back_square = current_square.copy()
        back_square.flip(axis=reflect_axis.get_start() - reflect_axis.get_end())
        self.play(Transform(current_square, back_square), Write(ref_rule), run_time=1.3)

        self.play(Write(K_conclusion), run_time=1.1)
        self.wait(1.2)

        self.play(
            FadeOut(reflect_axis),
            FadeOut(reflect_title),
            FadeOut(K_ref),
            FadeOut(ref_rule),
            FadeOut(K_conclusion),
            run_time=1.2,
        )

        # 不是任意子集都能成为子群：缺少封闭性。
        bad_title = Text("但不是所有子集都是子群", font_size=32)
        bad_set = MathTex(
            r"A=\{I,R_{\frac{\pi}{2}},F_0\}",
            font_size=38,
        )
        bad_product = MathTex(
            r"R_{\frac{\pi}{2}}\circ F_0=F_{\frac{\pi}{4}}\notin A",
            font_size=38,
        )
        bad_conclusion = MathTex(r"A \nleq D_4", font_size=44, color=c5)
        bad_panel = VGroup(bad_title, bad_set, bad_product, bad_conclusion).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT
        )
        bad_panel.move_to(RIGHT * right_x + UP * 0.25)

        self.play(Write(bad_title), Write(bad_set), run_time=1.8)
        self.play(Write(bad_product), run_time=2.0)
        self.play(Write(bad_conclusion), run_time=1.1)
        self.wait(2)

        self.wait(2)


class SubgroupPartitionTheorem(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("子群分割定理", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.add(title)

        subtitle = Text("用子群去“平移”，会把整个群分成不重叠的小块", font_size=30)
        subtitle.next_to(title, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(Write(subtitle), run_time=2)

        elems = [
            r"I",
            r"R_{\pi}",
            r"R_{\frac{\pi}{2}}",
            r"R_{\frac{3\pi}{2}}",
            r"F_0",
            r"F_{\frac{\pi}{2}}",
            r"F_{\frac{\pi}{4}}",
            r"F_{\frac{3\pi}{4}}",
        ]

        def element_card(tex, fill_color=BLUE_E):
            box = RoundedRectangle(
                width=1.55,
                height=0.72,
                corner_radius=0.08,
                stroke_width=2,
                stroke_color=WHITE,
                fill_color=fill_color,
                fill_opacity=0.35,
            )
            label = MathTex(tex, font_size=30)
            label.move_to(box)
            return VGroup(box, label)

        cards = VGroup(*[element_card(tex) for tex in elems]).arrange(RIGHT, buff=0.14)
        cards.scale(0.72)
        cards.move_to(UP * 0.9)

        G_label = MathTex(r"D_4", font_size=38)
        G_label.next_to(cards, LEFT, buff=0.25)

        self.play(
            Write(G_label),
            LaggedStart(*[FadeIn(card) for card in cards], lag_ratio=0.08),
            run_time=2,
        )
        self.wait(0.5)

        H_title = Text("先取一个最简单的子群", font_size=32)
        H_tex = MathTex(r"H=\{I,R_{\pi}\}", font_size=42)
        H_group = VGroup(H_title, H_tex).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        H_group.move_to(LEFT * 3.6 + DOWN * 1.1)

        coset_box_buff = 0.04
        H_box = SurroundingRectangle(
            VGroup(cards[0][0], cards[1][0]),
            color=c2,
            buff=coset_box_buff,
            corner_radius=0.08,
            stroke_width=4,
        )
        self.play(Write(H_group), Create(H_box), run_time=1.8)
        self.wait(1)

        coset_def = MathTex(
            r"gH=\{g\circ h\mid h\in H\}",
            font_size=40,
        )
        coset_def.move_to(RIGHT * 2.15 + DOWN * 1.1)
        self.play(Write(coset_def), run_time=1.8)
        self.wait(0.8)

        self.play(FadeOut(H_box), run_time=0.6)

        cosets = [
            (r"IH", [0, 1], c2),
            (r"R_{\frac{\pi}{2}}H", [2, 3], c3),
            (r"F_0H", [4, 5], c4),
            (r"F_{\frac{\pi}{4}}H", [6, 7], c5),
        ]

        coset_boxes = VGroup()
        coset_labels = VGroup()
        for name, indices, color in cosets:
            box = SurroundingRectangle(
                VGroup(*[cards[i][0] for i in indices]),
                color=color,
                buff=coset_box_buff,
                corner_radius=0.08,
                stroke_width=4,
            )
            label = MathTex(name, font_size=31, color=color)
            label.next_to(box, DOWN, buff=0.16)
            coset_boxes.add(box)
            coset_labels.add(label)

        for box, label in zip(coset_boxes, coset_labels):
            self.play(Create(box), Write(label), run_time=1.0)
            self.wait(0.3)

        self.play(
            FadeOut(H_group),
            FadeOut(coset_def),
            run_time=0.6,
        )

        notes = VGroup(
            Text("每块 2 个元素", font_size=24),
            Text("互不重叠", font_size=24),
            Text("刚好覆盖整个 D4", font_size=24),
        ).arrange(RIGHT, buff=0.62)
        notes.next_to(coset_labels, DOWN, buff=0.55)
        notes.shift(LEFT * 0.15)

        self.play(Write(notes[0]), run_time=1.2)
        self.play(Write(notes[1]), run_time=1.2)
        self.play(Write(notes[2]), run_time=1.2)
        self.wait(1.2)

        theorem = Text(
            "子群的元素个数，能够整除整个群的元素个数",
            font_size=31,
            color=c2,
        )
        theorem.move_to(DOWN * 1.55)

        self.play(
            FadeOut(notes),
            run_time=0.8,
        )
        self.play(Write(theorem), run_time=1.8)
        self.wait(2)
