# === Animation 2: Architekturansicht und Verarbeitung eines Eingabe-MNIST-Bildes durch das CNN ===
from pathlib import Path

import manim as m
import numpy as np
from PIL import Image


BACKGROUND = "#111111"
PLATE = "#182129"
GRID = "#788794"
CYAN = "#53D9EF"
ORANGE = "#F78654"
VIOLET = "#C1A3F7"
GOLD = "#FFD66B"

ARRAY_HEAD = 8
ARRAY_INDICES = (*range(ARRAY_HEAD), -2, -1)
DEMO_LOGITS = (-0.8, -0.4, -0.6, 0.3, -0.9, 2.4, 0.85, -0.2, -0.5, 0.1)

DIGIT = Path(__file__).with_name("mnist_digit.png")
X = np.linspace(-6.18, 6.18, 9)
Y = 0.65
SIZES = {28: 1.72, 14: 1.26, 7: 0.91}
TILT = np.array([[0.53, 0, 0], [0.28, 1, 0], [0, 0, 1]])
DEPTH = np.array([0.032, 0.020, 0])
LEVELS = 7

MOVE_TIME = 0.66
SCAN_TIME = 0.28
SAMPLE_COUNT = 5
STEP_PAUSE = 0.5
END_PAUSE = 3.0
STEPS = ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2",
         "flatten", "classifier")
LABELS = ("Eingabe", "Conv 1", "ReLU 1", "Pool 1", "Conv 2", "ReLU 2",
          "Pool 2", "Flatten", "Linear")
COLORS = (m.WHITE, CYAN, VIOLET, ORANGE, CYAN, VIOLET, ORANGE, m.WHITE, CYAN)


def demo_maps(image):
    """Berechnet illustrative Aktivierungen mit festen Beispielgewichten."""
    rng = np.random.default_rng(17)

    def conv(x, count):
        # Padding 1, Stride 1. Jeder Filter nutzt alle Eingangskanäle.
        kernels = rng.normal(0, 1 / np.sqrt(9 * len(x)), (count, len(x), 3, 3))
        if len(x) == 1:
            # Der erste sichtbare Filter betont vertikale Kanten.
            kernels[0, 0] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        return np.einsum("ochw,cijhw->oij", kernels, windows)

    def pool(x):
        c, h, w = x.shape
        return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))

    x = image[None, :, :]
    c1 = conv(x, 8)
    r1 = np.maximum(c1, 0)
    p1 = pool(r1)
    c2 = conv(p1, 16)
    r2 = np.maximum(c2, 0)
    p2 = pool(r2)
    return x, c1, r1, p1, c2, r2, p2


def text_at(text, size, x, y, color=m.WHITE):
    # Manim wählt die Standardschrift; es wird keine Schrift angegeben.
    return m.Text(text, font_size=size, color=color).move_to([x, y, 0])


def stage_label(index):
    label = m.VGroup(text_at(LABELS[index], 25, X[index], -0.76, COLORS[index]))
    extra = "28×28 px" if index == 0 else None
    if extra:
        label.add(text_at(extra, 16, X[index], -1.04, GRID))
    return label


class FeatureMap:
    """Ein Stapel mit einer sichtbar eingefärbten vorderen Merkmalskarte."""

    def __init__(self, tensor, index, scale):
        self.values = tensor[0]
        self.count = self.values.shape[0]
        self.side = SIZES[self.count]
        self.center = np.array([X[index], Y, 0]) - (len(tensor) - 1) / 2 * DEPTH
        self.scale = scale
        self.is_input = index == 0
        self.tensor = tensor

        boards = m.VGroup()
        # Hinten anfangen, damit jeweils die vordere Karte die nächste abdeckt.
        for channel in reversed(range(len(tensor))):
            board = m.Square(side_length=self.side, stroke_color=COLORS[index],
                             stroke_width=1.15, fill_color=PLATE, fill_opacity=1)
            boards.add(board.apply_matrix(TILT).shift(self.center + channel * DEPTH))

        grid = m.VGroup()
        for i in range(1, self.count):
            grid.add(m.Line(self.point(i, 0), self.point(i, self.count)),
                     m.Line(self.point(0, i), self.point(self.count, i)))
        grid.set_stroke(GRID, width=0.5, opacity=0.35).set_z_index(1)

        self.base = m.VGroup(boards, grid)
        self.ink = self.texture(self.values)
        self.body = m.VGroup(self.base, self.ink)
        self.label = stage_label(index)
        self.visual = m.VGroup(self.body, self.label)

    def point(self, row, col):
        p = np.array([col / self.count - 0.5, 0.5 - row / self.count, 0])
        return self.center + TILT @ (self.side * p)

    def corners(self, row, col, span=1):
        return [self.point(row, col), self.point(row + span, col),
                self.point(row + span, col + span), self.point(row, col + span)]

    def tile(self, row, col, span, color, opacity, stroke=0):
        return m.Polygon(*self.corners(row, col, span), stroke_color=color,
                         stroke_width=stroke, fill_color=color,
                         fill_opacity=opacity).set_z_index(2)

    def texture(self, values, negative_only=False):
        """Bündelt gleich helle Pixel in Pfade, damit der Zoom flüssig rendert."""
        paths = {}
        for r, c in np.argwhere(values != 0):
            value = values[r, c]
            if negative_only and value >= 0:
                continue
            level = max(1, min(LEVELS, round(abs(value) / self.scale * LEVELS)))
            key = (value < 0, level)
            if key not in paths:
                color = ORANGE if value < 0 else m.WHITE if self.is_input else CYAN
                paths[key] = m.VMobject(
                    stroke_width=0, fill_color=color, fill_opacity=level / LEVELS,
                )
            a, b, d, e = self.corners(r, c)
            paths[key].start_new_path(a)
            paths[key].add_points_as_corners([b, d, e, a])
        return m.VGroup(*paths.values())


class CNNArchitecture(m.MovingCameraScene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        frame = self.camera.frame
        frame.save_state()

        with Image.open(DIGIT) as image:
            pixels = np.asarray(image.convert("L"), dtype=float) / 255
        if pixels.shape != (28, 28):
            raise ValueError("mnist_digit.png muss 28 × 28 Pixel groß sein.")

        tensors = demo_maps(pixels)
        # Conv, ReLU und Pool teilen sich die Helligkeitsskala ihres Blocks.
        levels = (1.0, *([max(abs(tensors[1]).max(), 1e-9)] * 3),
                  *([max(abs(tensors[4]).max(), 1e-9)] * 3))
        maps = [FeatureMap(t, i, levels[i]) for i, t in enumerate(tensors)]

        title = text_at("Convolutional Neural Network", 46, 0, 3.25)
        forward = text_at("forward(x)", 27, 0, 2.55, CYAN)
        digit = maps[0].body.copy().shift(-maps[0].center)
        digit.apply_matrix(np.linalg.inv(TILT)).scale(3.25 / SIZES[28])
        self.play(m.FadeIn(title), m.FadeIn(digit))
        self.wait(0.45)
        self.play(m.Transform(digit, maps[0].body), m.FadeIn(maps[0].label),
                  m.FadeIn(forward), run_time=1.0)
        self.remove(digit, maps[0].label)
        self.add(maps[0].visual)

        array, boxes = self.array_row(maps[-1])
        outputs, dots = self.logit_row()
        stages = [*(card.visual for card in maps), array, outputs]

        code = None
        for i, name in enumerate(STEPS, 1):
            midpoint = (X[i - 1] + X[i]) / 2
            current = text_at(f"x = self.{name}(x)", 17, midpoint, -1.25, COLORS[i])
            width, y = (6.05, 0.28) if i <= 3 else (5.65, 0.18)
            animations = [
                frame.animate.set(width=width).move_to([midpoint, y, 0]),
                m.FadeIn(current),
            ]
            if code is not None:
                animations.extend([m.FadeOut(code), m.FadeOut(stages[i - 2])])
            self.play(*animations, run_time=MOVE_TIME)
            code = current

            if name.startswith(("conv", "pool")):
                self.scan(maps[i - 1], maps[i], pooling=name.startswith("pool"))
            elif name.startswith("relu"):
                self.relu(maps[i - 1], maps[i])
            elif name == "flatten":
                self.flatten(maps[-1], array, boxes)
            else:
                self.classifier(boxes, outputs, dots)
            self.wait(STEP_PAUSE)

        code, score_panel, highlight, prediction = self.select_prediction(outputs, dots, code)

        # Nach der letzten Nahaufnahme wieder alle Stationen auf eine Folie holen.
        overview_links = m.VGroup(*(
            m.Line(stages[i][0].get_right(), stages[i + 1][0].get_left(),
                   stroke_width=1.25, stroke_color=GRID).set_z_index(-2)
            for i in range(7)
        ))
        self.play(
            m.Restore(frame), m.FadeOut(code),
            m.FadeOut(score_panel), m.FadeOut(highlight),
            *(m.FadeIn(stage) for stage in stages[:-2]),
            m.FadeIn(overview_links),
            m.FadeIn(text_at(f"Vorhersage: {prediction}", 34, 0, -2.20, GOLD)),
            run_time=1.30,
        )
        self.wait(END_PAUSE)

    def scan(self, source, target, pooling):
        self.play(m.FadeIn(target.base), m.FadeIn(target.label), run_time=0.35)
        # Fünf tatsächlich benachbarte Fenster in einer aktiven Bildzeile
        # zeigen das Prinzip. Die restlichen Pixel werden danach eingeblendet.
        inner = 0 if pooling else 1
        values = abs(target.values)
        if inner:
            values = values[1:-1, 1:-1]
        runs = np.lib.stride_tricks.sliding_window_view(values, SAMPLE_COUNT, axis=1)
        r, c = np.unravel_index(np.argmax(runs.sum(axis=-1)), runs.shape[:2])
        r, c = r + inner, c + inner
        size, stride = (2, 2) if pooling else (3, 1)
        color = ORANGE if pooling else CYAN

        samples = m.VGroup()
        window = None
        for col in range(c, c + SAMPLE_COUNT):
            row_in, col_in = r * stride - inner, col * stride - inner
            next_window = source.tile(row_in, col_in, size, color, 0.10, 2.7)
            self.play(m.FadeIn(next_window) if window is None
                      else m.ReplacementTransform(window, next_window), run_time=SCAN_TIME)
            window = next_window
            value = target.values[r, col]
            shade = min(1, 0.15 + 0.85 * abs(value) / target.scale)
            result = target.tile(r, col, 1, ORANGE if value < 0 else CYAN, shade)
            samples.add(result)

            if pooling:
                patch = source.values[row_in:row_in + 2, col_in:col_in + 2]
                dr, dc = np.unravel_index(patch.argmax(), patch.shape)
                winner = source.tile(row_in + dr, col_in + dc, 1, m.WHITE, 0.63, 1)
                self.add(winner)
                self.play(m.TransformFromCopy(winner, result), run_time=0.24)
                self.remove(winner)
            else:
                pulse = m.Line(window.get_center(), result.get_center(), color=CYAN,
                               stroke_width=2.5).set_z_index(3)
                self.play(m.ShowPassingFlash(pulse, time_width=0.55),
                          m.FadeIn(result), run_time=0.24)

        self.play(m.FadeOut(window), m.FadeOut(samples),
                  m.FadeIn(target.ink), run_time=0.40)
        self.finish_map(target)

    def relu(self, source, target):
        # Dieselben positiven Werte bleiben stehen; der orange Anteil
        # verschwindet und wird dadurch auf der Karte zu null.
        negative = target.texture(source.values, negative_only=True)
        self.play(m.FadeIn(target.base), m.FadeIn(target.label),
                  m.FadeIn(target.ink), m.FadeIn(negative), run_time=0.48)
        self.wait(0.25)
        self.play(m.FadeOut(negative), run_time=0.60)
        self.finish_map(target)

    def finish_map(self, target):
        self.remove(target.base, target.ink, target.label)
        self.add(target.visual)

    @staticmethod
    def array_row(source):
        values = source.tensor.ravel()
        boxes = m.VGroup(*(
            m.Square(
                side_length=0.112,
                stroke_width=1.05,
                stroke_color=m.WHITE,
                fill_color=m.interpolate_color(
                    m.ManimColor(PLATE), m.ManimColor(CYAN),
                    min(1, values[i] / source.scale),
                ),
                fill_opacity=1,
            )
            for i in ARRAY_INDICES
        ))
        gap = m.Text("…", font_size=11, color=GRID)
        row = m.VGroup(*boxes[:ARRAY_HEAD], gap, *boxes[ARRAY_HEAD:])
        row.arrange(m.RIGHT, buff=0.025).move_to([X[7], Y, 0])
        return m.VGroup(row, stage_label(7)), boxes

    def flatten(self, source, array, boxes):
        n = source.count
        start = []
        for index in ARRAY_INDICES:
            # Erst zeilenweise durch eine Karte, dann zum nächsten Kanal.
            channel, offset = divmod(index % source.tensor.size, n * n)
            row, col = divmod(offset, n)
            start.append(
                source.tile(row, col, 1, CYAN, 0.65, 1).shift(channel * DEPTH)
            )

        gap = array[0][ARRAY_HEAD]
        self.play(*(m.FadeIn(cell) for cell in start), run_time=0.25)
        self.play(
            m.LaggedStart(
                *(m.ReplacementTransform(cell, box) for cell, box in zip(start, boxes)),
                lag_ratio=0.12,
            ),
            m.FadeIn(gap), m.FadeIn(array[1]), run_time=1.35,
        )
        self.remove(*boxes, gap, array[1])
        self.add(array)

    @staticmethod
    def logit_row():
        dots = m.VGroup(*(m.Dot(radius=0.044, color=CYAN) for _ in range(10)))
        dots.arrange(m.DOWN, buff=0.063).move_to([X[8] - 0.13, Y, 0])
        names = m.VGroup(*(m.Text(str(i), font_size=12).next_to(dot, m.RIGHT, buff=0.12)
                          for i, dot in enumerate(dots)))
        return m.VGroup(m.VGroup(dots, names), stage_label(8)), dots

    def classifier(self, boxes, output, dots):
        # Ausgewählte Kanten des dichten Klassifikators
        waves = []
        for i, dot in enumerate(dots):
            incoming = []
            for j in (i % len(boxes), (i + 5) % len(boxes)):
                # Start oben am Array, damit die Bahnen nicht durch Kästchen laufen.
                edge = m.CubicBezier(
                    boxes[j].get_top(), boxes[j].get_top() + 0.31 * m.UP,
                    dot.get_center() - 0.43 * m.RIGHT, dot.get_left(),
                    stroke_color=CYAN if j % 2 else ORANGE,
                    stroke_width=1.15, stroke_opacity=0.46,
                ).set_z_index(-1)
                flash = edge.copy().set_stroke(width=3.0, opacity=1)
                incoming.extend([m.Create(edge), m.ShowPassingFlash(flash, time_width=0.55)])
            waves.append(m.AnimationGroup(
                m.AnimationGroup(*incoming),
                m.FadeIn(dot), m.FadeIn(output[0][1][i]), lag_ratio=0.72,
            ))
        self.play(
            m.LaggedStart(*waves, lag_ratio=0.16),
            m.FadeIn(output[1]),
            # Die Kamera folgt der Signalrichtung bis zur Ausgabeschicht.
            self.camera.frame.animate.scale(0.97).shift(0.06 * m.RIGHT + 0.02 * m.DOWN),
            run_time=2.25,
        )
        self.remove(*dots, *output[0][1], output[1])
        self.add(output)

    def select_prediction(self, output, dots, code):
        names = output[0][1]
        score_x = X[-1] + 0.75
        scores = m.VGroup(*(
            text_at(f"{value:+.2f}".replace(".", ","), 11, score_x, dot.get_y())
            for value, dot in zip(DEMO_LOGITS, dots)
        ))
        heading = text_at(
            "Beispiel-Logits", 10, score_x, dots[0].get_y() + 0.20, GRID,
        )
        panel = m.VGroup(heading, scores)
        self.play(m.FadeIn(panel), run_time=0.42)

        winner = int(np.argmax(DEMO_LOGITS))
        argmax = text_at("x.argmax(dim=1)", 17, code.get_x(), code.get_y(), GOLD)
        winning_row = m.VGroup(dots[winner], names[winner], scores[winner])
        highlight = m.RoundedRectangle(
            width=winning_row.width + 0.10, height=0.14, corner_radius=0.025,
            stroke_color=GOLD, stroke_width=1.5,
        ).move_to(winning_row)
        self.play(
            m.ReplacementTransform(code, argmax), m.Create(highlight),
            *(item.animate.set_color(GOLD) for item in winning_row),
            run_time=0.72,
        )
        self.wait(1.75)
        return argmax, panel, highlight, winner