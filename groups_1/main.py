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
        ctex = TexTemplateLibrary.ctex
        group_def = MathTex(
            r"\text{群 } (G, \circ) \text{ 是一个集合 } G \text{ 和一个二元运算 } \circ \text{ 的组合，满足：}",
            font_size=32,
            tex_template=ctex,
        )
        group_def.to_edge(LEFT, buff=0.5).to_edge(UP, buff=1.5)
        self.play(Write(group_def), run_time=3)
        # 群的四条公理
        axioms = VGroup(
            MathTex(
                r"1.\ \text{封闭性：} \forall a, b \in G, a \circ b \in G",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"2.\ \text{结合性：} \forall a, b, c \in G, (a \circ b) \circ c = a \circ (b \circ c)",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"3.\ \text{单位元：} \exists e \in G, \forall a \in G, e \circ a = a \circ e = a",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"4.\ \text{逆元：} \forall a \in G, \exists b \in G, a \circ b = b \circ a = e",
                font_size=32,
                tex_template=ctex,
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
        square = Square(side_length=2, stroke_width=3)
        square.set_fill(BLUE_E, opacity=0.5)
        square.to_edge(LEFT, buff=1)
        self.play(Create(square), run_time=2)

        square_move = square.copy()
        # square_move.next_to(square, DOWN, buff=1.5)
        # self.play(Create(square_move), run_time=2)
        self.add(square_move)

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
            MathTex(
                r"1.\ \text{封闭性：} \forall a, b \in G, a \circ b \in G",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"2.\ \text{结合性：} \forall a, b, c \in G, (a \circ b) \circ c = a \circ (b \circ c)",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"3.\ \text{单位元：} \exists e \in G, \forall a \in G, e \circ a = a \circ e = a",
                font_size=32,
                tex_template=ctex,
            ),
            MathTex(
                r"4.\ \text{逆元：} \forall a \in G, \exists b \in G, a \circ b = b \circ a = e",
                font_size=32,
                tex_template=ctex,
            ),
        )
        for i, axiom in enumerate(axioms):
            axiom.next_to(Gtex, DOWN, buff=0.3 + i * 0.6, aligned_edge=LEFT)
            self.play(Write(axiom), run_time=2)
            self.wait(1)
        self.wait(2)


class ExampleofGroups(Scene):
    def construct(self):
        logo = Logo()
        self.add(logo.to_edge(DR, buff=0.5))

        title = Text("群论入门", font_size=48, color=WHITE).to_edge(UL, buff=0.5)
        self.play(Write(title))
