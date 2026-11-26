# === Animation 4: Aktivierungsinjektion in das MLP ===

import manim as m

from cnn_architecture import BACKGROUND, CYAN, GOLD, GRID, ORANGE, VIOLET, text_at


# Aufgezeichneter Einzeltest mit hidden, Position [Batch 0, Neuron 0].
ORIGINAL = "4,42"
INJECTED = "1000"
REFERENCE_CLASS = 5
INJECTED_CLASS = 6

HIDDEN_X, RELU_X, OUTPUT_X = -3.65, 0, 3.65
ROW_Y = 0.52
ROW_STEP = 0.17
END_PAUSE = 3.0


class MLPActivationInjection(m.Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        title = text_at("Aktivierungsinjektion im MLP", 44, 0, 3.12)
        hidden = self.column(HIDDEN_X, 12)
        relu = self.column(RELU_X, 12)
        output = self.column(OUTPUT_X, 10, CYAN)

        labels = m.VGroup(
            text_at("Linear · 12", 25, HIDDEN_X, -0.86, CYAN),
            text_at("ReLU", 25, RELU_X, -0.86, VIOLET),
            text_at("Linear · 10", 25, OUTPUT_X, -0.86, CYAN),
        )
        digits = m.VGroup(*(
            m.Text(str(i), font_size=13).next_to(dot, m.RIGHT, buff=0.16)
            for i, dot in enumerate(output)
        ))
        relu_links = m.VGroup(*(
            m.Line(left.get_right(), right.get_left(), stroke_color=VIOLET,
                   stroke_width=1.3, stroke_opacity=0.43)
            for left, right in zip(hidden, relu)
        ))
        # Einige Verbindungen stehen stellvertretend für das dichte Linear.
        output_links = m.VGroup()
        affected_links = []
        for i, target in enumerate(output):
            for start, color in ((relu[0], ORANGE), (relu[i + 1], CYAN)):
                edge = m.CubicBezier(
                    start.get_right(), start.get_right() + 0.85 * m.RIGHT,
                    target.get_left() - 0.85 * m.RIGHT, target.get_left(),
                    stroke_color=color, stroke_width=1.45, stroke_opacity=0.42,
                )
                output_links.add(edge)
                if start is relu[0]:
                    affected_links.append(edge)

        wires = m.VGroup(relu_links, output_links).set_z_index(-1)
        scene = m.VGroup(title, hidden, relu, output, labels, digits)
        self.play(m.FadeIn(wires), m.FadeIn(scene), run_time=0.80)

        reference = self.result("Ohne Injektion", REFERENCE_CLASS, -2.25)
        self.play(m.FadeIn(reference),
                  output[REFERENCE_CLASS].animate.set_color(GOLD),
                  digits[REFERENCE_CLASS].animate.set_color(GOLD), run_time=0.55)
        self.wait(0.65)

        position = text_at("hidden [0, 0]", 18, HIDDEN_X, 2.02, GRID)
        before = text_at(ORIGINAL, 30, HIDDEN_X, 1.69, CYAN)
        ring = m.Circle(radius=0.115, color=ORANGE, stroke_width=2).move_to(hidden[0])
        self.play(m.FadeIn(position), m.FadeIn(before), m.Create(ring), run_time=0.55)
        self.wait(0.42)

        after = text_at(f"{ORIGINAL} → {INJECTED}", 30, HIDDEN_X, 1.69, ORANGE)
        self.play(
            m.ReplacementTransform(before, after),
            hidden[0].animate.set_color(ORANGE),
            output[REFERENCE_CLASS].animate.set_color(CYAN),
            digits[REFERENCE_CLASS].animate.set_color(m.WHITE), run_time=0.55,
        )
        # ReLU lässt den positiven Ersatzwert 1000 unverändert.
        passed = text_at(INJECTED, 30, RELU_X, 1.69, ORANGE)
        self.play(self.pulse(relu_links[0]), m.FadeIn(passed),
                  relu[0].animate.set_color(ORANGE), run_time=0.65)
        self.play(*(self.pulse(edge) for edge in affected_links), run_time=0.85)

        injected = self.result("Mit Injektion", INJECTED_CLASS, 2.25)
        self.play(m.FadeIn(injected),
                  output[INJECTED_CLASS].animate.set_color(GOLD),
                  digits[INJECTED_CLASS].animate.set_color(GOLD), run_time=0.60)
        self.wait(END_PAUSE)

    @staticmethod
    def column(x, count, color=CYAN):
        return m.VGroup(*(
            m.Dot([x, ROW_Y + ((count - 1) / 2 - i) * ROW_STEP, 0],
                  radius=0.053, color=color)
            for i in range(count)
        ))

    @staticmethod
    def result(label, digit, x):
        return m.VGroup(text_at(label, 22, x, -1.83, m.WHITE),
                        text_at(str(digit), 42, x, -2.46, GOLD))

    @staticmethod
    def pulse(edge):
        glow = edge.copy().set_stroke(color=ORANGE, width=3.8, opacity=1)
        return m.ShowPassingFlash(glow.set_z_index(2), time_width=0.55)
