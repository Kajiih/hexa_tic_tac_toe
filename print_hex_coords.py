def get_coord_hex(radius: int) -> str:
    lines = []
    # Maximum width of a coordinate
    # For R<10, it's 2 chars (e.g. "-9", " 9")
    # Let's dynamically find max width
    max_val = radius
    cell_width = max(2, len(str(max_val)) + 1)  # e.g. " 9", "-9" -> length 2.

    # We want format string, e.g. "{:>2}"
    format_string = f"{{:>{cell_width}}}"

    for r in range(-(radius - 1), radius):
        q_start = max(-(radius - 1), -(radius - 1) - r)
        q_end = min((radius - 1), (radius - 1) - r)

        # Calculate indent.
        # In a standard hex grid, the center row has no indent.
        # The rows above and below have an indent of (abs(r) * some_factor)
        # Since each cell takes 'cell_width' + 1 (for space) chars...
        # Let's say cell is 2 chars. Plus 1 space = 3 chars per cell.
        # Indent should be abs(r) * (cell_width + 1) / 2?
        # A simple way is to pad with spaces:
        # standard text hex: indent = " " * abs(r)

        # But our cells are wider.
        # For a symmetrical stagger, the offset per row from center is half a cell width.
        # But text characters are discrete.
        # If cell_width=2 and separator is 1 space (total 3), offsetting by 1.5 chars is impossible.
        # Let's use cell_width=3, separator=1 space (total 4). Half is 2 chars.

        # Let's try cell_width = 3 for everything.
        # cell_str: e.g. "  0" or " -1"
        # separator: " "
        # row indent: abs(r) * 2 spaces (which is exactly half of a 4-char step)

        indent = " " * (abs(r) * 2)

        q_line = []
        r_line = []
        for q in range(q_start, q_end + 1):
            q_line.append(f"{q:>3}")
            r_line.append(f"{r:>3}")

        lines.append(indent + " ".join(q_line))
        lines.append(indent + " ".join(r_line))
        lines.append("")  # empty line between rows? No, let's keep it tight.

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_coord_hex(3))
