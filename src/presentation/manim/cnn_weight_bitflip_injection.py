# === Animation 5: Weight Bitflip im CNN ===
import struct
import manim as m

from cnn_architecture import BACKGROUND, CYAN, GRID, ORANGE, PLATE, text_at


# Position und Bitnummer kommen aus weight_bitflip_injection.py.
# Der Zahlenwert ist ein Beispiel und stammt nicht aus dem Checkpoint.
EXAMPLE_WEIGHT = 0.125
BIT = 30
END_PAUSE = 3.0

PIXELS = (
    "0011100",
    "0001100",
    "0001100",
    "0011000",
    "0110000",
)


def bitflip_example():
    """Berechnet die Bits vor und nach dem Flip sowie den neuen Zahlenwert."""
    bits, = struct.unpack("<I", struct.pack("<f", EXAMPLE_WEIGHT))
    flipped = bits ^ (1 << BIT)
    value, = struct.unpack("<f", struct.pack("<I", flipped))
    return f"{bits:032b}", f"{flipped:032b}", value


def bitstrip(bits):
    """Ordnet die 32 Bits in Vorzeichen, Exponent und Mantisse."""
    cells = []
    digits = []
    groups = m.VGroup()

    for part, color in ((bits[:1], GRID), (bits[1:9], CYAN), (bits[9:], GRID)):
        block = m.VGroup()
        for bit in part:
            cell = m.Square(
                side_length=0.285, stroke_color=color,
                stroke_width=1, fill_color=PLATE, fill_opacity=1,
            )
            digit = m.Text(bit, font_size=14).move_to(cell)
            cells.append(cell)
            digits.append(digit)
            block.add(m.VGroup(cell, digit))
        block.arrange(m.RIGHT, buff=0.025)
        groups.add(block)

    groups.arrange(m.RIGHT, buff=0.30).move_to([0, 0.19, 0])
    labels = m.VGroup(*(
        text_at(label, 16, block.get_x(), 0.70, color)
        for label, block, color in zip(
            ("Vorzeichen", "Exponent", "Mantisse"), groups, (GRID, CYAN, GRID)
        )
    ))
    return m.VGroup(groups, labels), cells, digits


def kernel(center, side, value):
    """Zeichnet den 3×3-Kernel und hebt das Gewicht oben links hervor."""
    grid = m.VGroup()
    for i in range(9):
        row, col = divmod(i, 3)
        cell = m.Square(
            side_length=side,
            stroke_color=ORANGE if i == 0 else GRID,
            stroke_width=2.0 if i == 0 else 1.2,
            fill_color="#4A2C21" if i == 0 else PLATE,
            fill_opacity=1,
        ).move_to([center[0] + (col - 1) * side, center[1] + (1 - row) * side, 0])
        marker = m.Text(
            value if i == 0 else "·", font_size=15,
            color=ORANGE if i == 0 else GRID,
        ).move_to(cell)
        grid.add(m.VGroup(cell, marker))
    return grid


class CNNWeightBitflipInjection(m.Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        original_bits, flipped_bits, result = bitflip_example()
        index = 31 - BIT  # Bit 30 ist die zweite Stelle von links.

        value_before = str(EXAMPLE_WEIGHT).replace(".", ",")
        mantissa, exponent = f"{result:.2e}".split("e")
        superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        value_after = (
            f"{mantissa.replace('.', ',')} × 10"
            f"{str(int(exponent)).translate(superscripts)}"
        )

        # 1. Das betroffene Gewicht im ersten Filter zeigen.
        title = text_at("Bitflip in einem CNN-Gewicht", 44, 0, 3.18)
        weights = kernel((0, 0.35), 0.83, value_before)
        path = text_at("conv1.weight[0, 0, 0, 0]", 25, 0, -1.22)
        illustration = text_at("Beispielgewicht", 19, 0, -1.92, GRID)
        self.play(
            m.FadeIn(title), m.FadeIn(weights),
            m.FadeIn(path), m.FadeIn(illustration), run_time=0.75,
        )
        self.wait(2.0)

        # 2. Die Float32-Darstellung zeigen und genau Bit 30 umlegen.
        decimal = text_at(value_before, 34, 0, 1.52)
        strip, cells, digits = bitstrip(original_bits)
        self.play(
            m.FadeOut(weights),
            m.FadeOut(path), m.FadeOut(illustration),
            m.FadeIn(decimal), m.FadeIn(strip), run_time=0.72,
        )

        target = cells[index]
        pointer = m.Arrow(
            target.get_bottom() + 0.57 * m.DOWN,
            target.get_bottom() + 0.035 * m.DOWN,
            buff=0, color=ORANGE, stroke_width=2.7,
            max_tip_length_to_length_ratio=0.24,
        )
        bit_name = text_at("Bit 30", 20, target.get_x(), -0.78, ORANGE)
        self.play(
            m.GrowArrow(pointer), m.FadeIn(bit_name),
            target.animate.set_stroke(ORANGE, width=2.5), run_time=0.54,
        )
        self.wait(2.0)

        new_bit = text_at(flipped_bits[index], 14, target.get_x(), target.get_y(), ORANGE)
        decimal_change = text_at(f"{value_before}  →  ≈ {value_after}", 34, 0, 1.52, ORANGE)
        explanation = text_at("Ein Bit im Exponenten ändert sich.", 22, 0, -1.60)
        self.play(
            m.Transform(digits[index], new_bit),
            target.animate.set_fill("#4A2C21", opacity=1),
            m.ReplacementTransform(decimal, decimal_change),
            m.FadeIn(explanation), run_time=0.64,
        )
        self.wait(2.25)

        # 3. Dasselbe veränderte Gewicht wird an mehreren Positionen benutzt.
        compact = text_at(f"{value_before}  →  ≈ {value_after}", 28, 0, 2.24, ORANGE)
        changed_kernel = kernel((-3.35, 0.0), 0.47, "w′")
        kernel_label = text_at("Veränderter Kernel", 19, -3.35, -1.03, ORANGE)
        image, pixel_grid = self.pixel_grid()
        relation = m.Arrow(
            changed_kernel.get_right() + 0.22 * m.RIGHT,
            image.get_left() + 0.19 * m.LEFT,
            buff=0, color=GRID, stroke_width=2.4,
        )
        frame = self.window(pixel_grid, 0, 0)
        self.play(
            m.ReplacementTransform(decimal_change, compact),
            m.FadeOut(strip), m.FadeOut(bit_name),
            m.FadeOut(pointer), m.FadeOut(explanation),
            m.FadeIn(changed_kernel), m.FadeIn(kernel_label),
            m.GrowArrow(relation), m.FadeIn(image), m.FadeIn(frame), run_time=0.90,
        )
        for row, col in ((0, 1), (1, 2), (2, 4)):
            self.wait(0.25)
            self.play(m.Transform(frame, self.window(pixel_grid, row, col)), run_time=0.45)

        self.wait(END_PAUSE)

    @staticmethod
    def pixel_grid():
        side = 0.37
        cells = [
            [
                m.Square(
                    side_length=side, stroke_color=GRID, stroke_width=0.8,
                    fill_color="#DCE7EE" if pixel == "1" else PLATE, fill_opacity=1,
                ).move_to([2.05 + (c - 3) * side, (2 - r) * side, 0])
                for c, pixel in enumerate(row)
            ]
            for r, row in enumerate(PIXELS)
        ]
        return m.VGroup(*(cell for row in cells for cell in row)), cells

    @staticmethod
    def window(cells, row, col):
        region = m.VGroup(*(
            cells[r][c]
            for r in range(row, row + 3)
            for c in range(col, col + 3)
        ))
        outline = m.SurroundingRectangle(region, color=ORANGE, buff=0, stroke_width=2.5)
        first = cells[row][col].copy().set_stroke(ORANGE, width=2.3)
        first.set_fill(ORANGE, opacity=0.52)
        return m.VGroup(outline, first)