def build():
    # Newport 55lb-class trolling motor toroidal propeller (mm).
    # Axis along Z; hub base on the XY floor (z=0).
    # Factory-like hub OD (~90 mm), airfoil blades, short bottom drive-pin groove.
    # Blades: loft of thin airfoil polylines; chord pitched in r–z / +θ sense.

    tip_r = 133.0
    hub_r = 45.0  # ~90 mm OD, typical 55 lb factory prop hub
    hub_h = 60.0
    bore_r = 4.8  # ~3/8 in shaft
    flange_r = hub_r + 4.0

    # Drive pin (~Ø5 mm): short pocket around the shaft, not through flange OD
    pin_d = 5.0
    pin_clear = 0.5
    groove_w = pin_d + pin_clear
    pin_seat_z = 16.0
    groove_h = pin_seat_z + pin_d * 0.55
    groove_half = hub_r - 10.0
    groove_mask_r = hub_r - 6.0

    n_loops = 3
    root_inset = 5.0
    n_sec = 40

    # Thin airfoil: chord ~50, thickness ~8.2% chord
    chord = 50.0
    thick = 4.1
    pitch_deg = 32.0  # foil face pitch; LE faces +θ

    z_lead = groove_h + 3.0
    z_trail = z_lead + 36.0
    z_tip_boost = 8.0

    def path_point(t):
        # Monotonic loop; flattened tip apex eases curvature vs section chord
        span = 2.0 * math.pi / n_loops
        theta = span * t
        r_root = hub_r - root_inset
        tip_blend = 1.0 - abs(2.0 * t - 1.0) ** 2.6
        r = r_root + (tip_r - r_root) * tip_blend
        z = z_lead + (z_trail - z_lead) * t + z_tip_boost * math.sin(math.pi * t)
        z = max(z_lead - 1.0, min(hub_h - 3.0, z))
        return (r * math.cos(theta), r * math.sin(theta), z)

    def vsub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def vadd(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def vscale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def vdot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def vcross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def vnorm(a):
        L = math.sqrt(vdot(a, a))
        if L < 1e-12:
            return (1.0, 0.0, 0.0)
        return (a[0] / L, a[1] / L, a[2] / L)

    def tangent_at(t):
        dt = 1.0 / (n_sec * 4.0)
        t0 = max(0.0, t - dt)
        t1 = min(1.0, t + dt)
        return vnorm(vsub(path_point(t1), path_point(t0)))

    def section_size(t):
        # Hard taper on t∈[0,0.2]∪[0.8,1]
        edge = 0.20

        def ease(u):
            return u * u * (3.0 - 2.0 * u)

        if t < edge:
            w = ease(t / edge)
        elif t > 1.0 - edge:
            w = ease((1.0 - t) / edge)
        else:
            w = 1.0
        c = chord * (0.38 + 0.62 * w)
        th = thick * (0.50 + 0.50 * w)
        return c, th

    def airfoil_pts(c, th, n=20):
        # Local 2D: x along chord (LE at -c/2, sharp TE at +c/2), y thickness
        peak = math.sqrt(1.0 / 3.0) * (2.0 / 3.0)
        upper = []
        for i in range(n + 1):
            u = i / float(n)
            x = -0.5 * c + u * c
            env = math.sqrt(max(u, 1e-9)) * (1.0 - u) / peak
            y = 0.5 * th * env
            if i == n:
                y = 0.0
            upper.append((x, y))
        lower = []
        for i in range(n - 1, 0, -1):
            u = i / float(n)
            x = -0.5 * c + u * c
            env = math.sqrt(max(u, 1e-9)) * (1.0 - u) / peak
            y = 0.5 * th * env
            lower.append((x, -y))
        return upper + lower

    def chord_xdir(p, T):
        # Pitched chord in r–z / +θ sense: LE toward +θ, face visible (not plan-view tube)
        rxy = math.hypot(p[0], p[1])
        if rxy < 1e-6:
            theta_hat = (0.0, 1.0, 0.0)
        else:
            theta_hat = (-p[1] / rxy, p[0] / rxy, 0.0)
        up = (0.0, 0.0, 1.0)
        a = pitch_deg * math.pi / 180.0
        # xDir points LE→TE; LE toward +θ ⇒ xDir ≈ -θ·cos(a) + z·sin(a)
        pref = vadd(vscale(theta_hat, -math.cos(a)), vscale(up, math.sin(a)))
        xdir = vsub(pref, vscale(T, vdot(pref, T)))
        if math.sqrt(vdot(xdir, xdir)) < 0.25:
            # Fallback: radial×T gives thickness-ish; use up projected
            xdir = vsub(up, vscale(T, vdot(up, T)))
        return vnorm(xdir)

    def make_loop():
        wires = []
        prev_x = None
        for i in range(n_sec):
            t = i / float(n_sec - 1)
            p = path_point(t)
            T = tangent_at(t)
            xdir = chord_xdir(p, T)
            # Prevent 180° section flips that crease the loft
            if prev_x is not None and vdot(xdir, prev_x) < 0.0:
                xdir = vscale(xdir, -1.0)
            prev_x = xdir
            c, th = section_size(t)
            pts2 = airfoil_pts(c, th)
            # Plane(origin, xDir=chord, normal=path tangent);
            # thickness (plane Y) = T×xDir ≈ loop-plane normal
            wire = (
                cq.Workplane(cq.Plane(p, xdir, T))
                .polyline(pts2)
                .close()
                .wires()
                .val()
            )
            wires.append(wire)

        solid = cq.Solid.makeLoft(wires)
        return cq.Workplane("XY").newObject([solid])

    hub = cq.Workplane("XY").circle(hub_r).extrude(hub_h)
    hub = hub.edges(">Z").fillet(2.0)

    nose = (
        cq.Workplane("XY")
        .workplane(offset=hub_h)
        .circle(hub_r * 0.72)
        .extrude(6.0)
    )
    flange = cq.Workplane("XY").circle(flange_r).extrude(3.5)
    hub = hub.union(nose).union(flange)

    prop = hub
    step = 360.0 / n_loops
    for i in range(n_loops):
        blade = make_loop().rotate((0, 0, 0), (0, 0, 1), i * step)
        prop = prop.union(blade)

    bore = cq.Workplane("XY").circle(bore_r).extrude(hub_h + 10.0)
    prop = prop.cut(bore)

    # Short drive-pin pocket (unchanged)
    hub_mask = cq.Workplane("XY").circle(groove_mask_r).extrude(groove_h + pin_d)

    groove = (
        cq.Workplane("XY")
        .rect(2.0 * groove_half, groove_w)
        .extrude(groove_h)
        .intersect(hub_mask)
        .rotate((0, 0, 0), (0, 0, 1), 30.0)
    )
    prop = prop.cut(groove)

    seat = (
        cq.Workplane("XZ")
        .center(0, pin_seat_z)
        .circle(groove_w * 0.5)
        .extrude(groove_half, both=True)
        .intersect(hub_mask)
        .rotate((0, 0, 0), (0, 0, 1), 30.0)
    )
    prop = prop.cut(seat)

    return prop
