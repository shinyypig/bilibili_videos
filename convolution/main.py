from manim import *


class ContinuousConvolution(Scene):
    def construct(self):
        # Define the functions f(t) and g(t)
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
