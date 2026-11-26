# === Animation 3: Architekturansicht und Verarbeitung eines Eingabe-MNIST-Bildes durch das MLP ===
import manim as m
import numpy as np
from PIL import Image

# Die Szene liegt neben cnn_architecture.py und nutzt dessen Farben,
# Standardschrift und kippbares Pixelbild.
from cnn_architecture import (
    BACKGROUND, CYAN, DIGIT, GOLD, GRID, ORANGE, PLATE, TILT, VIOLET,
    FeatureMap, text_at,
)

Y = 0.65
FLAT_X, HIDDEN_X, RELU_X, OUTPUT_X = -3.65, -0.95, 1.72, 4.45
ARRAY_HEAD = 8
ARRAY_INDICES = (*range(ARRAY_HEAD), -2, -1)
CAMERA_WIDTH = 6.25
MOVE_TIME = 0.75
STEP_PAUSE = 0.70
END_PAUSE = 3.0

# Beispielwerte für die Präsentation; die Animation lädt keinen Checkpoint.
HIDDEN_VALUES = (-0.6, 1.4, -0.2, 0.8, -0.9, 0.4, -0.5, 1.1, 0.3, -0.4, 0.7, -0.3)
LOGITS = (-0.4, 0.2, -0.7, 0.5, -0.3, 2.1, 0.8, -0.1, 0.1, -0.5)


class MLPArchitecture(m.MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        frame = self.camera.frame
        frame.save_state()

        with Image.open(DIGIT) as image:
            pixels = np.asarray(image.convert("L"), dtype=float) / 255
        if pixels.shape != (28, 28):
            raise ValueError("mnist_digit.png muss 28 x 28 Pixel groß sein.")

        # FeatureMap mit genau einem Kanal: das gleiche Bild wie im CNN-Clip.
        source = FeatureMap(pixels[None], 0, 1.0)
        input_x = source.center[0]
        title = text_at("Multilayer Perceptron", 44, 0, 3.25)
        forward = text_at("forward(x)", 27, 0, 2.55, CYAN)
        front = source.body.copy().shift(-source.center)
        front.apply_matrix(np.linalg.inv(TILT)).scale(3.25 / source.side)

        self.play(m.FadeIn(title), m.FadeIn(front))
        self.wait(0.5)
        self.play(
            m.ReplacementTransform(front, source.body),
            m.FadeIn(source.label), m.FadeIn(forward), run_time=1.0,
        )

        # 28 × 28 → 784: Die sichtbaren Kästchen sind Anfang und Ende des Arrays.
        code = self.focus(input_x, FLAT_X, "x = self.flatten(x)", m.WHITE)
        boxes = m.VGroup(*(
            m.Square(side_length=0.14, stroke_color=m.WHITE, stroke_width=1.05,
                     fill_color=m.interpolate_color(m.ManimColor(PLATE), m.WHITE, pixels.flat[i]),
                     fill_opacity=1)
            for i in ARRAY_INDICES
        ))
        gap = m.Text("…", font_size=12, color=GRID)
        row = m.VGroup(*boxes[:ARRAY_HEAD], gap, *boxes[ARRAY_HEAD:])
        row.arrange(m.RIGHT, buff=0.035).move_to([FLAT_X, Y, 0])
        marked = [
            source.tile(*divmod(i % pixels.size, 28), 1, CYAN, 0.6, 1)
            for i in ARRAY_INDICES
        ]
        self.play(*(m.FadeIn(cell) for cell in marked), run_time=0.28)
        self.play(
            m.LaggedStart(*(m.ReplacementTransform(cell, box) for cell, box in zip(marked, boxes)),
                          lag_ratio=0.12),
            m.FadeIn(gap), m.FadeIn(self.label("Flatten", FLAT_X)), run_time=1.30,
        )
        self.wait(STEP_PAUSE)

        # 784 → 12: Einige Lichtbahnen stehen für die dichte Verbindung.
        code = self.focus(FLAT_X, HIDDEN_X, "x = self.hidden(x)", CYAN, code)
        hidden, hidden_dots, hidden_values = self.column(HIDDEN_X, HIDDEN_VALUES)
        hidden_values.shift(0.63 * m.LEFT + 0.08 * m.UP)
        examples = text_at("Beispielwerte", 11, HIDDEN_X - 0.28, 1.85, GRID)
        first_wires = self.flow(boxes, hidden_dots, from_boxes=True, reveal=(
            m.FadeIn(hidden), m.FadeIn(examples), m.FadeIn(self.label("Linear · 12", HIDDEN_X, CYAN)),
        ))
        self.wait(STEP_PAUSE)

        # 12 → 12: Beim Kopieren bleiben die positiven Werte stehen.
        # Für jeden negativen Eintrag ändern sich anschließend nur Wert und Farbe.
        code = self.focus(HIDDEN_X, RELU_X, "x = self.relu(x)", VIOLET, code)
        before, before_dots, _ = self.column(RELU_X, HIDDEN_VALUES)
        after, relu_dots, relu_values = self.column(RELU_X, np.maximum(HIDDEN_VALUES, 0))

        # Ein Eingang pro Ausgang: ReLU mischt keine Neuronen miteinander.
        relu_wires = m.VGroup(*(
            m.Line(
                left.get_right(), right.get_left(),
                stroke_color=VIOLET, stroke_opacity=0.35, stroke_width=1.05,
            )
            for left, right in zip(hidden_dots, before_dots)
        )).set_z_index(-1)
        self.play(m.Create(relu_wires), m.TransformFromCopy(hidden, before),
                  m.FadeIn(self.label("ReLU", RELU_X, VIOLET)), run_time=0.80)
        self.wait(0.35)
        self.play(m.ReplacementTransform(before, after), run_time=0.80)
        self.wait(STEP_PAUSE)

        # 12 → 10: Die zweite lineare Schicht liefert je einen Score pro Ziffer.
        code = self.focus(RELU_X, OUTPUT_X, "x = self.classifier(x)", CYAN, code,
                          hide=(relu_values))
        _, out_dots, scores = self.column(OUTPUT_X, LOGITS, logits=True)
        digits = m.VGroup(*(
            m.Text(str(i), font_size=12).next_to(dot, m.RIGHT, buff=0.11)
            for i, dot in enumerate(out_dots)
        ))
        score_heading = text_at("Beispiel-Logits", 11, OUTPUT_X + 0.83, 1.71, GRID)
        self.flow(relu_dots, out_dots, reveal=(
            m.FadeIn(out_dots), m.FadeIn(digits), m.FadeIn(scores),
            m.FadeIn(score_heading), m.FadeIn(self.label("Linear · 10", OUTPUT_X, CYAN)),
        ))
        self.wait(STEP_PAUSE)

        # Wie beim CNN: Erst nach Linear wählt argmax den größten Logit.
        winner = int(np.argmax(LOGITS))
        selected = m.VGroup(out_dots[winner], digits[winner], scores[winner])
        ring = m.RoundedRectangle(
            width=selected.width + 0.10, height=0.15, corner_radius=0.025,
            stroke_color=GOLD, stroke_width=1.5,
        ).move_to(selected)
        argmax = text_at("x.argmax(dim=1)", 18, code.get_x(), code.get_y(), GOLD)
        self.play(m.ReplacementTransform(code, argmax), m.Create(ring),
                  *(part.animate.set_color(GOLD) for part in selected), run_time=0.75)
        self.wait(1.25)

        prediction = text_at(f"Vorhersage: {winner}", 34, 0, -2.20, GOLD)
        input_link = m.Line(source.body.get_right(), row.get_left(),
                            stroke_color=GRID, stroke_width=1.25).set_z_index(-1)
        self.play(
            m.Restore(frame), m.FadeOut(argmax), m.FadeOut(ring), m.FadeOut(scores),
            m.FadeOut(score_heading), m.FadeOut(examples),
            m.FadeOut(hidden_values),
            m.FadeIn(input_link),
            m.FadeIn(prediction), run_time=1.3,
        )
        self.wait(END_PAUSE)

    @staticmethod
    def label(name, x, color=m.WHITE):
        return text_at(name, 25, x, -0.76, color)

    def focus(self, left_x, right_x, line, color, previous=None, *, hide=()):
        """Die Kamera zeigt immer Eingang und Ausgang der aktuellen Operation."""
        middle = (left_x + right_x) / 2
        code = text_at(line, 18, middle, -1.28, color)
        moves = [self.camera.frame.animate.set(width=CAMERA_WIDTH).move_to([middle, 0.25, 0]),
                 m.FadeIn(code)]
        if previous is not None:
            moves.append(m.FadeOut(previous))
        moves.extend(m.FadeOut(item) for item in hide)
        self.play(*moves, run_time=MOVE_TIME)
        return code

    @staticmethod
    def column(x, values, logits=False):
        dots = m.VGroup()
        numbers = m.VGroup()
        for i, value in enumerate(values):
            y = Y + ((len(values) - 1) / 2 - i) * 0.17
            color = CYAN if logits or value > 0 else ORANGE if value < 0 else GRID
            dot = m.Dot([x, y, 0], radius=0.045, color=color)
            if not logits and value == 0:
                dot.set_fill(opacity=0).set_stroke(GRID, width=1.5)
            digits = f"{value:+.1f}".replace(".", ",") if value != 0 else "0"
            dots.add(dot)
            numbers.add(text_at(digits, 11, x + (0.83 if logits else 0.35), y,
                                m.WHITE if logits else color))
        return m.VGroup(dots, numbers), dots, numbers

    def flow(self, source, target, *, reveal, from_boxes=False):
        """Zeigt exemplarische Kanten; ein Lichtimpuls läuft entlang jeder Kante."""
        edges = m.VGroup()
        pulses = []
        for i, end in enumerate(target):
            for j in (i % len(source), (i + len(source) // 2) % len(source)):
                begin = source[j]
                color = CYAN if (i + j + (i // 3)) % 3 else ORANGE
                start = begin.get_top() if from_boxes else begin.get_right()
                edge = m.CubicBezier(start, start + (0.42 * m.UP if from_boxes else 0.55 * m.RIGHT),
                                     end.get_center() - 0.55 * m.RIGHT, end.get_left(),
                                     stroke_color=color, stroke_width=1.25, stroke_opacity=0.41)
                edge.set_z_index(-1)
                edges.add(edge)
                pulse = edge.copy().set_stroke(width=3.1, opacity=1)
                pulses.append(m.ShowPassingFlash(pulse, time_width=0.55))

        self.play(
            m.Create(edges, lag_ratio=0.05), m.LaggedStart(*pulses, lag_ratio=0.05),
            *reveal, self.camera.frame.animate.shift(0.04 * m.RIGHT), run_time=1.85,
        )
        return edges