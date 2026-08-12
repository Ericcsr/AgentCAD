def build():
    """Simple parametric chair used by mock LLM backend."""
    SEAT_W, SEAT_D, SEAT_T, SEAT_H = 450.0, 420.0, 28.0, 450.0
    LEG_W, LEG_INSET = 38.0, 25.0
    BACK_H, BACK_T, BACK_RECLINE = 420.0, 22.0, 8.0
    STRETCHER_T, STRETCHER_H = 22.0, 120.0
    SLATS, GAP = 4, 18.0

    half_w = SEAT_W / 2 - LEG_INSET - LEG_W / 2
    half_d = SEAT_D / 2 - LEG_INSET - LEG_W / 2
    span_w = 2 * half_w - LEG_W
    span_d = 2 * half_d - LEG_W

    def union(parts):
        s = parts[0]
        for p in parts[1:]:
            s = s.union(p)
        return s

    parts = []
    parts.append(
        cq.Workplane("XY")
        .workplane(offset=SEAT_H - SEAT_T)
        .box(SEAT_W, SEAT_D, SEAT_T, centered=(True, True, False))
        .edges("|Z").fillet(6)
    )
    for x, y in [(-half_w, -half_d), (half_w, -half_d), (-half_w, half_d), (half_w, half_d)]:
        parts.append(
            cq.Workplane("XY").box(LEG_W, LEG_W, SEAT_H, centered=(True, True, False)).translate((x, y, 0))
        )
    for x in (-half_w, half_w):
        parts.append(
            cq.Workplane("XY")
            .workplane(offset=STRETCHER_H - STRETCHER_T / 2)
            .box(STRETCHER_T, span_d, STRETCHER_T, centered=(True, True, False))
            .translate((x, 0, 0))
        )
    parts.append(
        cq.Workplane("XY")
        .workplane(offset=STRETCHER_H - STRETCHER_T / 2)
        .box(span_w, STRETCHER_T, STRETCHER_T, centered=(True, True, False))
        .translate((0, -half_d, 0))
    )

    back = []
    for x in (-half_w, half_w):
        back.append(
            cq.Workplane("XY")
            .workplane(offset=SEAT_H)
            .box(LEG_W, LEG_W, BACK_H, centered=(True, True, False))
            .translate((x, half_d, 0))
        )
    back.append(
        cq.Workplane("XY")
        .workplane(offset=SEAT_H + BACK_H - BACK_T)
        .box(span_w + LEG_W, BACK_T, BACK_T, centered=(True, True, False))
        .translate((0, half_d, 0))
    )
    slat_w = (span_w - (SLATS - 1) * GAP) / SLATS
    start_x = -span_w / 2 + slat_w / 2
    for i in range(SLATS):
        x = start_x + i * (slat_w + GAP)
        back.append(
            cq.Workplane("XY")
            .workplane(offset=SEAT_H + 5)
            .box(slat_w, BACK_T * 0.7, BACK_H - BACK_T - 10, centered=(True, True, False))
            .translate((x, half_d, 0))
        )
    backrest = union(back).rotate((0, half_d, SEAT_H), (1, half_d, SEAT_H), BACK_RECLINE)
    return union(parts).union(backrest)
