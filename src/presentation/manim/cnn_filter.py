# === Animation 1: Schematische Ausschnitt eines CNN-Filters + Injektion ===

import manim as m


# Farben und Layout
BACKGROUND_COLOR = "#111111"
DARK_PIXEL_COLOR = "#1E2836"
LIGHT_PIXEL_COLOR = m.WHITE
GRID_COLOR = m.GRAY
WINDOW_COLOR = m.YELLOW
POSITIVE_COLOR = m.BLUE_B
NEGATIVE_COLOR = m.PURPLE_B
ZERO_COLOR = m.GRAY_A

CELL_SIZE = 0.54
CELL_GAP = 0.04
IMAGE_X, FILTER_X, FEATURE_MAP_X = -4.2, 0, 4.2
LABEL_Y = 2.15

# Zeiten in Sekunden
MOVE_TIME = 0.45
VALUE_FADE_TIME = 0.25
POSITION_PAUSE = 0.35
END_PAUSE = 2.0

# Schematischer Bildausschnitt und fester Kantenfilter
PIXELS = (
    (0, 0, 1, 1, 0, 0),
    (0, 0, 1, 1, 0, 0),
    (0, 0, 1, 1, 0, 0),
    (0, 1, 1, 0, 0, 0),
    (0, 1, 1, 0, 0, 0),
    (1, 1, 0, 0, 0, 0),
)
KERNEL = (
    (-1, 0, 1),
    (-1, 0, 1),
    (-1, 0, 1),
)


def make_grid(size, x):
    """Erstellt ein quadratisches Raster mit zeilenweise angeordneten Zellen."""
    squares = m.VGroup(*[
        m.Square(side_length=CELL_SIZE, stroke_width=1, color=GRID_COLOR)
        for _ in range(size * size)
    ])
    squares.arrange_in_grid(rows=size, cols=size, buff=CELL_GAP)
    return squares.move_to([x, 0, 0])


class CNNFilter(m.Scene):
    def construct(self):
        self.camera.background_color = m.ManimColor(BACKGROUND_COLOR)

        image_size = len(PIXELS)
        kernel_size = len(KERNEL)
        # Schrittweite 1, ohne Padding: 6 - 3 + 1 = 4 Positionen pro Achse.
        output_size = image_size - kernel_size + 1
        radius = kernel_size // 2

        image = make_grid(image_size, IMAGE_X)
        feature_map = make_grid(output_size, FEATURE_MAP_X)

        for row, pixel_row in enumerate(PIXELS):
            for col, pixel in enumerate(pixel_row):
                color = LIGHT_PIXEL_COLOR if pixel else DARK_PIXEL_COLOR
                image[row * image_size + col].set_fill(color, opacity=1)

        title = m.Text(
            "Ein Filter erkennt lokale Muster", font_size=44
        ).to_edge(m.UP)
        labels = m.VGroup(*[
            m.Text(text, font_size=25).move_to([x, LABEL_Y, 0])
            for x, text in (
                (IMAGE_X, "Bildausschnitt"),
                (FILTER_X, "Ein 3 × 3-Filter"),
                (FEATURE_MAP_X, "Merkmalskarte"),
            )
        ])
        # Die sichtbaren Gewichte stammen aus derselben Matrix wie die Rechnung.
        kernel_text = "\n".join(
            "   ".join(f"{weight:+d}" if weight else "0" for weight in row)
            for row in KERNEL
        )
        weights = m.Text(kernel_text, font_size=30).move_to([FILTER_X, 0, 0])
        footer = m.Text(
            "Schematischer Bildausschnitt und Filter, ohne Padding",
            font_size=20,
            color=ZERO_COLOR,
        ).to_edge(m.DOWN)
        self.play(m.FadeIn(title, labels, image, weights, feature_map, footer))

        # Der Rahmen umfasst drei Zellen und die zwei Zwischenräume.
        window_size = kernel_size * CELL_SIZE + (kernel_size - 1) * CELL_GAP
        window = m.Square(
            side_length=window_size, color=WINDOW_COLOR, stroke_width=4
        )

        for row in range(output_size):
            for col in range(output_size):
                # (row, col) ist die linke obere Ecke des Fensters.
                # Für den gelben Rahmen brauchen wir dessen Mittelpunkt.
                center_index = (row + radius) * image_size + (col + radius)
                target = image[center_index].get_center()

                if row == 0 and col == 0:
                    window.move_to(target)
                    self.play(m.Create(window), run_time=MOVE_TIME)
                else:
                    self.play(
                        window.animate.move_to(target), run_time=MOVE_TIME
                    )

                # Pixel und Gewichte paarweise multiplizieren, dann aufsummieren.
                value = sum(
                    PIXELS[row + i][col + j] * KERNEL[i][j]
                    for i in range(kernel_size)
                    for j in range(kernel_size)
                )
                if value > 0:
                    color = POSITIVE_COLOR
                elif value < 0:
                    color = NEGATIVE_COLOR
                else:
                    color = ZERO_COLOR

                # Jede Fensterposition ergibt genau eine Zelle der Merkmalskarte.
                output_index = row * output_size + col
                number = m.Text(
                    f"{value:+d}" if value else "0", font_size=22, color=color
                ).move_to(feature_map[output_index])
                self.play(m.FadeIn(number), run_time=VALUE_FADE_TIME)
                self.wait(POSITION_PAUSE)

        self.wait(END_PAUSE)


# === Einstellungen für die Aktivierungsinjektion ===

FAULT_VALUE = 1000
FAULT_COLOR = m.ORANGE
FAULT_ROW, FAULT_COL = 2, 1  # Zeile und Spalte im schematischen 4×4-Raster.

COMPARISON_X = 3.2  # Abstand der beiden Raster von der Bildschirmmitte.
CAPTION_GAP = 0.30
VALUE_FONT_SIZE = 22

ZOOM_SCALE = 0.42  # Kleinerer Bildausschnitt = stärkerer Zoom.
ZOOM_TIME = 1.0

REVEAL_TIME = 0.8
CHANGE_TIME = 0.65
COMPARE_PAUSE = 2.0
PRE_INJECTION_PAUSE = 1.0


class CNNActivationInjection(m.MovingCameraScene):
    def construct(self):
        self.camera.background_color = m.ManimColor(BACKGROUND_COLOR)

        kernel_size = len(KERNEL)
        output_size = len(PIXELS) - kernel_size + 1
        fault_index = FAULT_ROW * output_size + FAULT_COL

        # Beide Läufe beginnen mit derselben Merkmalskarte aus der ersten Szene.
        reference_grid = make_grid(output_size, -COMPARISON_X)
        faulty_grid = make_grid(output_size, COMPARISON_X)
        reference_numbers = m.VGroup()
        faulty_numbers = m.VGroup()

        for row in range(output_size):
            for col in range(output_size):
                # Dieselbe Filterrechnung wie in CNNFilter, ohne Padding.
                value = sum(
                    PIXELS[row + i][col + j] * KERNEL[i][j]
                    for i in range(kernel_size)
                    for j in range(kernel_size)
                )
                index = row * output_size + col

                if value > 0:
                    color = POSITIVE_COLOR
                elif value < 0:
                    color = NEGATIVE_COLOR
                else:
                    color = ZERO_COLOR

                number = m.Text(
                    f"{value:+d}" if value else "0",
                    font_size=VALUE_FONT_SIZE,
                    color=color,
                )
                reference_numbers.add(number.move_to(reference_grid[index]))
                faulty_numbers.add(number.copy().move_to(faulty_grid[index]))

                if index == fault_index:
                    original_value = value

        title = m.Text(
            "Fehler in einer Aktivierung", font_size=44
        ).to_edge(m.UP)

        reference_label = m.Text("Referenzlauf", font_size=28).next_to(
            reference_grid, m.UP, buff=CAPTION_GAP
        )
        faulty_label = m.Text("Fehlerlauf", font_size=28).next_to(
            faulty_grid, m.UP, buff=CAPTION_GAP
        )

        self.play(
            m.FadeIn(title, reference_label, faulty_label),
            m.FadeIn(reference_grid, reference_numbers),
            m.FadeIn(faulty_grid, faulty_numbers),
            run_time=REVEAL_TIME,
        )
        self.wait(COMPARE_PAUSE)

        # Links und rechts dieselbe Zelle markieren. Nur rechts ändert sich der Wert.
        reference_frame = m.SurroundingRectangle(
            reference_grid[fault_index], color=POSITIVE_COLOR, buff=0
        )
        faulty_frame = m.SurroundingRectangle(
            faulty_grid[fault_index], color=FAULT_COLOR, buff=0
        )
        self.play(
            m.Create(reference_frame),
            m.Create(faulty_frame),
            run_time=CHANGE_TIME,
        )

        # Gesamtansicht merken und Raster samt Beschriftung heranzoomen.
        frame = self.camera.frame
        frame.save_state()
        focus = m.VGroup(faulty_grid, faulty_label)

        self.play(
            frame.animate.scale(ZOOM_SCALE).move_to(focus.get_center()),
            run_time=ZOOM_TIME,
        )
        self.wait(PRE_INJECTION_PAUSE)

        injected_number = m.Text(
            str(FAULT_VALUE), font_size=VALUE_FONT_SIZE, color=FAULT_COLOR
        )
        # Vier Ziffern sollen in die markierte Zelle passen.
        if injected_number.width > 0.85 * CELL_SIZE:
            injected_number.scale_to_fit_width(0.85 * CELL_SIZE)
        injected_number.move_to(faulty_grid[fault_index])

        self.play(
            m.Transform(faulty_numbers[fault_index], injected_number),
            run_time=CHANGE_TIME,
        )

        # Kurz auf der 1000 bleiben, dann beide Läufe im Endbild zeigen.
        self.wait(COMPARE_PAUSE)
        self.play(m.Restore(frame), run_time=ZOOM_TIME)
        self.wait(END_PAUSE)