def build():
    """Newport 55lb Large 3-blade trolling-motor propeller (mm)."""
    # Official Large 3-blade: 10.1" OD, 3.625" hub bore (55/62/86 lb)
    IN = 25.4
    DIA = 10.1 * IN
    HUB_ID = 3.625 * IN
    HUB_OD = 108.0
    HUB_H = 52.0
    CAVITY_DEPTH = 40.0
    SHAFT_D = 18.0
    NOSE_H = 10.0
    NOSE_OD = 72.0

    # Drive-pin groove in cavity ceiling: prop slides onto a pin already
    # in the shaft; pin nests in this diametral groove (no exterior hole)
    PIN_D = 6.5
    PIN_CLEAR = 0.6
    GROOVE_W = PIN_D + PIN_CLEAR
    GROOVE_DEPTH = 7.5
    GROOVE_LEN = 48.0

    # Wider blade chords (F6) with proportionally thicker airfoil sections
    ROOT_CHORD = 90.0
    TIP_CHORD = 50.0
    ROOT_T = 11.5
    TIP_T = 4.8
    ROOT_CAMBER = 3.6
    TIP_CAMBER = 1.4
    # Deep root overlap so blades fully fuse into hub wall before cavity cut
    ROOT_OVERLAP = 18.0
    PITCH_DEG = 24.0
    N_BLADES = 3
    FOIL_N = 18
    # Rounded tip cap: short radial zone ending in a blunt ellipse
    TIP_ROUND = 14.0
    END_CHORD = 16.0
    END_T = 3.2

    def union_all(parts):
        s = parts[0]
        for p in parts[1:]:
            s = s.union(p)
        return s

    def airfoil_pts(chord, tmax, camber, n):
        """Closed YZ section: chord along Y, thickness along Z, centered."""
        upper = []
        lower = []
        for i in range(n + 1):
            x = i / float(n)
            xt = x
            if xt < 1e-8:
                xt = 1e-8
            yt = 5.0 * tmax * (
                0.2969 * math.sqrt(xt)
                - 0.1260 * x
                - 0.3516 * x * x
                + 0.2843 * x * x * x
                - 0.1015 * x * x * x * x
            )
            if yt < 0.2:
                yt = 0.2
            yc = 4.0 * camber * x * (1.0 - x)
            y = (x - 0.5) * chord
            upper.append((y, yc + yt))
            lower.append((y, yc - yt))
        pts = []
        for p in upper:
            pts.append(p)
        i = len(lower) - 2
        while i > 0:
            pts.append(lower[i])
            i -= 1
        return pts

    # Solid hub + nose first (no cavity yet) so blades fuse fully into the wall
    hub = cq.Workplane("XY").circle(HUB_OD / 2.0).extrude(HUB_H)
    nose = (
        cq.Workplane("XY")
        .workplane(offset=HUB_H)
        .circle(NOSE_OD / 2.0)
        .extrude(NOSE_H)
    )
    hub = hub.union(nose)

    # Airfoil blades with rounded tips; roots start deep inside hub radius
    r0 = HUB_OD / 2.0 - ROOT_OVERLAP
    r1 = DIA / 2.0
    r_near = r1 - TIP_ROUND
    r_mid = r0 + 0.55 * (r_near - r0)
    z_mid = HUB_H * 0.55

    mid_chord = ROOT_CHORD + (TIP_CHORD - ROOT_CHORD) * 0.55
    mid_t = ROOT_T + (TIP_T - ROOT_T) * 0.55
    mid_camber = ROOT_CAMBER + (TIP_CAMBER - ROOT_CAMBER) * 0.55

    root_pts = airfoil_pts(ROOT_CHORD, ROOT_T, ROOT_CAMBER, FOIL_N)
    mid_pts = airfoil_pts(mid_chord, mid_t, mid_camber, FOIL_N)
    near_pts = airfoil_pts(TIP_CHORD, TIP_T, TIP_CAMBER, FOIL_N)

    blades = []
    for i in range(N_BLADES):
        ang = i * (360.0 / N_BLADES)
        blade = (
            cq.Workplane("YZ")
            .workplane(offset=r0)
            .polyline(root_pts)
            .close()
            .workplane(offset=r_mid - r0)
            .polyline(mid_pts)
            .close()
            .workplane(offset=r_near - r_mid)
            .polyline(near_pts)
            .close()
            .workplane(offset=r1 - r_near)
            .ellipse(END_CHORD / 2.0, END_T / 2.0)
            .loft()
        )
        blade = blade.translate((0.0, 0.0, z_mid))
        blade = blade.rotate((0.0, 0.0, z_mid), (1.0, 0.0, z_mid), PITCH_DEG)
        blade = blade.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), ang)
        blades.append(blade)

    # Fuse blades into solid hub so exteriors are fully connected (F7)
    prop = union_all([hub] + blades)

    # Cut cavity after union: removes blade roots that would show inside (F8)
    cavity = cq.Workplane("XY").circle(HUB_ID / 2.0).extrude(CAVITY_DEPTH)
    prop = prop.cut(cavity)

    # Prop-shaft through-hole (hub + nose)
    prop = prop.cut(
        cq.Workplane("XY").circle(SHAFT_D / 2.0).extrude(HUB_H + NOSE_H)
    )

    # Internal drive-pin groove
    groove = (
        cq.Workplane("XY")
        .workplane(offset=CAVITY_DEPTH - 0.2)
        .box(
            GROOVE_LEN,
            GROOVE_W,
            GROOVE_DEPTH + 0.2,
            centered=(True, True, False),
        )
    )
    prop = prop.cut(groove)

    return prop
