# 3D-printable ISO M10x1.5 hex bolt and matching internally-threaded nut.
# No stand: both parts sit on the XY floor (hex faces are the stable base).
# FDM clearance is built into the major diameters (bolt undersize, nut oversize).
# Units: mm. Z up.

PITCH = 1.5
BOLT_MAJOR = 9.70
BOLT_MINOR = 8.20
NUT_MAJOR = 10.40
NUT_MINOR = 8.50
HEAD_AF = 16.0
HEAD_H = 6.4
NUT_H = 8.4
THREAD_L = 28.0
SHOULDER_H = 1.6
TIP_CHAMFER = 1.4
NUT_CHAMFER = 1.0
_OVERLAP = 0.30


def _hex_prism(across_flats, height):
    across_corners = across_flats / math.cos(math.radians(30.0))
    return (
        cq.Workplane("XY")
        .polygon(6, across_corners)
        .extrude(height)
        .rotate((0, 0, 0), (0, 0, 1), 30.0)
    )


def _helix_path(pitch, height, radius):
    return cq.Workplane(obj=cq.Wire.makeHelix(pitch, height, radius))


def _trim_z(solid, d_ref, length):
    clip = cq.Workplane("XY").box(
        d_ref * 4.0, d_ref * 4.0, length, centered=(True, True, False)
    )
    return solid.intersect(clip)


def _external_thread(d_major, d_minor, pitch, length):
    """Right-hand male V-thread (triangular helix) trimmed to length."""
    r_out = d_major / 2.0
    r_in = d_minor / 2.0
    half = pitch * 0.48
    extra = pitch
    path = _helix_path(pitch, length + 2.0 * extra, r_in)
    profile = (
        cq.Workplane("XZ")
        .moveTo(r_in - _OVERLAP, -half)
        .lineTo(r_out, 0.0)
        .lineTo(r_in - _OVERLAP, half)
        .close()
    )
    swept = profile.sweep(path, isFrenet=True).translate((0.0, 0.0, -extra))
    return _trim_z(swept, d_major, length)


def _internal_thread_cutter(d_major, d_minor, pitch, length):
    """Fat trapezoidal helix that gouges a visible female thread into a nut wall."""
    r_out = d_major / 2.0
    r_in = d_minor / 2.0
    extra = pitch
    inner_w = pitch * 0.08
    outer_w = pitch * 0.47
    path = _helix_path(pitch, length + 2.0 * extra, r_in)
    profile = (
        cq.Workplane("XZ")
        .moveTo(r_in - _OVERLAP, -inner_w)
        .lineTo(r_in - _OVERLAP, inner_w)
        .lineTo(r_out + 0.20, outer_w)
        .lineTo(r_out + 0.20, -outer_w)
        .close()
    )
    swept = profile.sweep(path, isFrenet=True).translate((0.0, 0.0, -extra))
    return _trim_z(swept, d_major + 2.0, length)


def _cone(r1, r2, height, z0, zdir):
    return cq.Workplane(
        cq.Solid.makeCone(r1, r2, height, cq.Vector(0, 0, z0), cq.Vector(0, 0, zdir))
    )


def _bolt():
    head = _hex_prism(HEAD_AF, HEAD_H)
    shoulder = (
        cq.Workplane("XY")
        .workplane(offset=HEAD_H)
        .circle(BOLT_MAJOR / 2.0)
        .extrude(SHOULDER_H)
    )
    core_z = HEAD_H + SHOULDER_H
    core = (
        cq.Workplane("XY")
        .workplane(offset=core_z)
        .circle(BOLT_MINOR / 2.0)
        .extrude(THREAD_L)
    )
    thread = _external_thread(BOLT_MAJOR, BOLT_MINOR, PITCH, THREAD_L).translate(
        (0.0, 0.0, core_z)
    )
    bolt = head.union(shoulder).union(core).union(thread)

    z_tip = core_z + THREAD_L
    keep = _cone(
        BOLT_MAJOR / 2.0,
        max(BOLT_MAJOR / 2.0 - TIP_CHAMFER, 0.6),
        TIP_CHAMFER,
        z_tip - TIP_CHAMFER,
        1.0,
    )
    ring = (
        cq.Workplane("XY")
        .workplane(offset=z_tip - TIP_CHAMFER)
        .circle(BOLT_MAJOR / 2.0 + 4.0)
        .extrude(TIP_CHAMFER + 1.0)
        .cut(keep)
    )
    return bolt.cut(ring)


def _nut():
    body = _hex_prism(HEAD_AF, NUT_H)
    # Pilot smaller than the thread crest so the helical cutter must remove wall material.
    pilot = cq.Workplane("XY").circle(NUT_MINOR / 2.0 - 0.15).extrude(NUT_H)
    body = body.cut(pilot)
    body = body.cut(_internal_thread_cutter(NUT_MAJOR, NUT_MINOR, PITCH, NUT_H))

    # Short 45° lead-in only — previous full-radius cones erased the internal thread.
    r_face = NUT_MINOR / 2.0 + NUT_CHAMFER
    r_inner = NUT_MINOR / 2.0
    body = body.cut(_cone(r_face, r_inner, NUT_CHAMFER, NUT_H, -1.0))
    body = body.cut(_cone(r_face, r_inner, NUT_CHAMFER, 0.0, 1.0))
    return body


def parts():
    gap = 22.0
    return {
        "bolt": _bolt().translate((-gap, 0.0, 0.0)),
        "nut": _nut().translate((gap, 0.0, 0.0)),
    }


def build():
    items = list(parts().values())
    solid = items[0]
    for item in items[1:]:
        solid = solid.union(item)
    return solid
