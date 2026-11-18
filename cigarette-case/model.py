"""벽부형 담배 케이스 (섹터1: 담배갑 2개 / 섹터2: 라이터 1개)

좌표계 (단위 mm)
  x : 폭 (왼쪽 → 오른쪽)
  y : 깊이 (0 = 벽에 붙는 뒷면, + 방향 = 앞쪽)
  z : 높이 (0 = 바닥)

구성 부품
  body.stl : 본체 (뒷면을 베드에 두고 출력)
  door.stl : 하단 힌지식 앞문 + 독수리 부조 (뒷면을 베드에 두고 출력)
  힌지 핀  : 1.75 mm 필라멘트 토막 (또는 Ø2 mm 금속 핀)
  자석     : Ø6 × 2 mm 네오디뮴 4개 (본체 2, 문 2)

사용법
  python3 model.py                 # 기본(일반 담배갑) 프리셋
  python3 model.py --preset universal
"""

import argparse
import math
import os

import numpy as np
import manifold3d as m3d
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# 파라미터
# ---------------------------------------------------------------------------
PRESETS = {
    # 국제 표준 20개비 플립탑 갑 (예: 88 x 55 x 23 mm)
    "regular": dict(pack_w=55.0, pack_d=23.0, pack_h=88.0),
    # 일반 갑 + 슬림(100mm 개비) 갑까지 수용하는 범용 높이
    "universal": dict(pack_w=55.0, pack_d=23.0, pack_h=108.0),
}

LIGHTER_W, LIGHTER_D, LIGHTER_H = 25.0, 15.0, 82.0  # BIC 맥시급 라이터 수용

WALL = 2.4        # 측벽 (0.4 노즐 6라인)
BACK = 3.0        # 뒷판 (양면테이프 부착면)
TOP = 3.0         # 윗판
BOTTOM = 4.0      # 바닥판 (힌지 축 높이와 동일)
DIVIDER = 2.0     # 섹터 칸막이
DOOR_T = 4.0      # 문 두께
GAP = 0.4         # 문-프레임 유격

CLEAR_W = 4.0     # 담배갑 2개 폭 여유 (합계)
CLEAR_D = 3.0     # 깊이 여유
CLEAR_H = 5.0     # 높이 여유 (자석 보스 아래, 갑을 앞으로 기울여 꺼낼 때 여유 포함)

KNUCKLE_R = 4.0   # 힌지 너클 반경 (Ø8)
KNUCKLE_CUT = 4.5 # 상대 너클 회전 여유 반경
KNUCKLE_N = 5     # 너클 개수 (본체-문-본체-문-본체)
KNUCKLE_GAP = 0.4 # 너클 축방향 유격
PIN_D = 2.1       # 핀 구멍 (1.75 필라멘트용)

MAG_D, MAG_T = 6.3, 2.2  # Ø6x2 자석 포켓 (압입/접착)
BOSS_W, BOSS_H = 10.0, 9.0

SCREW_HOLES = True       # 테이프 + 나사 병행 고정용 접시머리 구멍
SCREW_D, SCREW_HEAD_D = 4.2, 8.5

SEG = 96  # 원 분할 수


# ---------------------------------------------------------------------------
# 기본 도형 헬퍼
# ---------------------------------------------------------------------------
def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_x(x0, x1, y, z, r):
    """x축 방향 원기둥"""
    c = m3d.Manifold.cylinder(x1 - x0, r, r, SEG)  # z축 방향으로 생성
    return c.rotate([0, 90, 0]).translate([x0, y, z])


def cyl_y(y0, y1, x, z, r, r2=None):
    """y축 방향 원기둥 (r→r2 원뿔 가능)"""
    r2 = r if r2 is None else r2
    c = m3d.Manifold.cylinder(y1 - y0, r, r2, SEG)
    return c.rotate([-90, 0, 0]).translate([x, y0, z])


def prism_x(x0, x1, pts_yz):
    """yz 평면 다각형을 x 방향으로 돌출"""
    cs = m3d.CrossSection([pts_yz])
    return cs.extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


# ---------------------------------------------------------------------------
# 독수리 부조 (높이맵 → 솔리드)
# ---------------------------------------------------------------------------
def eagle_heightmap(span=104.0, res=0.25):
    """양식화된 독수리 높이맵 (날개를 위로 펼친 정면 자세, 머리는 옆을 봄)

    설계 좌표(날개폭 약 128)로 그린 뒤 span(mm)에 맞게 축소한다.
    반환: xs, zs (mm, 중심 0), H (문 앞면 기준 높이 mm, 바탕은 음수)
    """
    design_span = 128.0
    k = span / design_span
    r = res / k                                     # 설계 좌표에서의 격자 간격
    xs_d = np.arange(-68, 68 + r, r)
    zs_d = np.arange(-40, 58 + r, r)
    X, Z = np.meshgrid(xs_d, zs_d)
    H = np.full(X.shape, -1.0)

    def dome(cx, cz, rx, rz, ang, h_edge, h_peak, ridge=0.0, taper=0.0):
        ca, sa = math.cos(ang), math.sin(ang)
        u = ((X - cx) * ca + (Z - cz) * sa) / rx
        v = (-(X - cx) * sa + (Z - cz) * ca) / rz
        if taper:  # 끝(u>0) 쪽으로 갈수록 폭이 좁아지는 깃털
            v = v / np.clip(1 - taper * np.clip(u, 0, 1), 0.25, 1)
        d2 = u * u + v * v
        prof = np.sqrt(np.clip(1.0 - d2, 0, 1))
        h = h_edge + (h_peak - h_edge) * prof
        if ridge:
            h = h + ridge * np.exp(-(v * rz / 0.45) ** 2) * np.clip(1 - np.abs(u), 0, 1)
        np.maximum(H, np.where(d2 < 1.0, h, -1.0), out=H)

    def feather(bx, bz, ang, length, width, h_edge, h_peak, ridge=0.22, taper=0.35):
        cx = bx + math.cos(ang) * length / 2
        cz = bz + math.sin(ang) * length / 2
        dome(cx, cz, length / 2, width / 2, ang, h_edge, h_peak, ridge, taper)

    lerp = lambda a, b, t: a + (b - a) * t

    for side in (1, -1):
        P = lambda x, z: (side * x, z)
        A = lambda deg: math.radians(deg) if side > 0 else math.pi - math.radians(deg)

        # 날개 골격(앞전): 어깨 → 손목 → 날개끝
        shoulder, wrist, tip = (7, 9), (30, 27), (44, 35)

        # 칼깃(첫째날개깃): 손목에서 부채꼴로 뻗는 긴 '손가락' 깃 7장
        for i in range(7):
            t = i / 6
            bx, bz = lerp(26, 41, t), lerp(24, 33, t)
            feather(*P(bx, bz), A(lerp(-22, 38, t)), lerp(24, 27, math.sin(t * math.pi)) + 3 * t,
                    5.6, 0.8, 1.5)
        # 둘째날개깃: 팔 아래로 늘어지는 깃 9장
        for i in range(9):
            t = i / 8
            bx, bz = lerp(8, 27, t), lerp(7, 23, t)
            feather(*P(bx, bz), A(lerp(-78, -30, t)), lerp(17, 23, t), 5.6, 0.9, 1.6)
        # 큰날개덮깃
        for i in range(10):
            t = i / 9
            bx, bz = lerp(8, 37, t), lerp(9, 31, t)
            feather(*P(bx, bz), A(lerp(-70, 8, t)), lerp(10, 13, t), 5.0, 1.7, 2.4)
        # 작은날개덮깃 (비늘)
        for i in range(9):
            t = i / 8
            bx, bz = lerp(7, 33, t), lerp(10, 31, t)
            feather(*P(bx, bz), A(lerp(-60, -5, t)), lerp(6, 7.5, t), 4.4, 2.4, 3.0, ridge=0, taper=0.15)
        # 앞전 융기
        for t in np.linspace(0, 1, 14):
            if t < 0.62:
                s = t / 0.62
                ax, az = lerp(shoulder[0], wrist[0], s), lerp(shoulder[1], wrist[1], s)
            else:
                s = (t - 0.62) / 0.38
                ax, az = lerp(wrist[0], tip[0], s), lerp(wrist[1], tip[1], s)
            dome(*P(ax, az + 1.0), 3.0, 2.3, A(35), 2.8, 3.3)

        # 꼬리깃 (가운데 1장 + 좌우 3장씩)
        for j, deg in enumerate((-90, -80, -70, -60)):
            if j == 0 and side < 0:
                continue
            feather(*P(0.8 * j, -12), A(deg), 19 - 1.2 * j, 5.4, 1.0, 1.9)
        # 다리(깃털 바지) + 발톱
        dome(*P(4.2, -11), 3.0, 4.6, A(-75), 2.5, 3.2)
        for j in range(3):
            feather(*P(4.6 + (j - 1) * 1.5, -15.5), A(-90 + (j - 1) * 25), 3.4, 1.3, 2.5, 2.9, 0, 0.5)

    # 몸통
    dome(0, -3, 7.2, 11.5, 0, 2.5, 3.8)
    dome(0, 5, 7.6, 7.5, 0, 2.9, 4.0)
    # 목 깃(들쭉날쭉한 가장자리)
    for j in range(7):
        a = math.radians(-150 + j * 20)
        feather(1.0, 12.0, a, 6.0, 3.0, 3.0, 3.6, 0, 0.5)
    # 머리 (오른쪽을 봄) + 이마 융기 + 갈고리 부리
    dome(1.6, 15.8, 4.8, 3.9, math.radians(-8), 3.3, 4.5)
    dome(3.4, 17.4, 2.6, 0.9, math.radians(-10), 4.0, 4.7)        # 눈썹뼈
    dome(6.6, 15.0, 2.8, 1.7, math.radians(-15), 3.4, 4.2)        # 윗부리
    dome(8.4, 13.6, 1.4, 0.9, math.radians(-70), 3.3, 3.9)        # 부리 갈고리
    # 가슴 비늘 무늬 (홈)
    for r_ in range(4):
        for c_ in range(-1, 2):
            cx, cz = c_ * 3.8 + (r_ % 2) * 1.9 - 0.95, 6 - r_ * 3.5
            dd = np.hypot(X - cx, Z - cz)
            groove = (np.abs(dd - 2.1) < 0.3) & (Z < cz) & (H > 2.6)
            H[groove] -= 0.35
    # 눈, 부리선
    H[np.hypot(X - 3.6, Z - 16.2) < 0.8] -= 0.55
    H[(np.abs((Z - 15.0) + 0.25 * (X - 6.0)) < 0.25) & (X > 5.0) & (X < 8.5)] -= 0.3

    H = gaussian_filter(H, sigma=0.22 / r)         # 약 0.22 mm 스무딩
    H = np.maximum(H, -0.6)                        # 바탕: 문 표면 아래로 묻힘
    H = gaussian_filter(H, sigma=0.12 / r)
    xs, zs = xs_d * k, zs_d * k
    # 높이도 살짝 축소 (최대 약 4 mm)
    return xs, zs, H * min(1.0, 0.9 + 0.1 * k)


def heightfield_solid(xs, zs, H, base=-0.6):
    """높이맵(H) 을 x-z 평면, 돌출 방향 +y 인 폐쇄 솔리드로 변환"""
    nz, nx = H.shape
    top = np.stack(np.meshgrid(xs, zs), -1).reshape(-1, 2)
    vt = np.column_stack([top[:, 0], H.reshape(-1), top[:, 1]])  # (x, y=h, z)
    vb = np.column_stack([top[:, 0], np.full(len(top), base - 0.4), top[:, 1]])
    verts = np.vstack([vt, vb]).astype(np.float32)
    off = len(vt)
    idx = np.arange(nx * nz).reshape(nz, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tri_top = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3),
                              np.stack([b, c, d], -1).reshape(-1, 3)])
    tri_bot = tri_top[:, ::-1] + off
    # 측면
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    nxt = np.roll(ring, -1)
    side = np.concatenate([np.stack([ring, nxt, nxt + off], -1),
                           np.stack([ring, nxt + off, ring + off], -1)])
    tris = np.concatenate([tri_top, tri_bot, side]).astype(np.uint32)
    mesh = m3d.Mesh(vert_properties=verts, tri_verts=tris)
    solid = m3d.Manifold(mesh)
    if solid.status() != m3d.Error.NoError:
        # 방향이 반대인 경우 뒤집기
        mesh = m3d.Mesh(vert_properties=verts, tri_verts=tris[:, ::-1].copy())
        solid = m3d.Manifold(mesh)
    return solid


# ---------------------------------------------------------------------------
# 모델 생성
# ---------------------------------------------------------------------------
def build(preset="regular"):
    p = PRESETS[preset]
    s1_w = 2 * p["pack_w"] + CLEAR_W                 # 섹터1 내폭
    s2_w = LIGHTER_W + 5.0                           # 섹터2 내폭
    in_d = p["pack_d"] + CLEAR_D                     # 내부 깊이
    in_h = max(p["pack_h"], LIGHTER_H) + CLEAR_H + BOSS_H  # 내부 높이

    W = WALL + s1_w + DIVIDER + s2_w + WALL
    D = BACK + in_d + DOOR_T                         # 앞면 y 좌표
    Ht = BOTTOM + in_h + TOP
    zi0, zi1 = BOTTOM, BOTTOM + in_h                 # 내부 z 범위
    xi0, xi1 = WALL, W - WALL
    yd = D - DOOR_T                                  # 문 뒷면 y
    axis_y, axis_z = D, BOTTOM                       # 힌지 축

    # 너클 구간 (본체 = 짝수, 문 = 홀수)
    seg_w = W / KNUCKLE_N
    segs = [(i * seg_w, (i + 1) * seg_w) for i in range(KNUCKLE_N)]

    # ----------------------------- 본체 -----------------------------------
    body = box(0, W, 0, D, 0, Ht) - box(xi0, xi1, BACK, D + 1, zi0, zi1)

    # 칸막이 (문 뒤 0.5 mm 까지)
    div_x0 = xi0 + s1_w
    body += box(div_x0, div_x0 + DIVIDER, BACK - 0.1, yd - 0.5, zi0 - 0.1, zi1 + 0.1)

    # 자석 보스 (좌상단, 우상단 모서리, 뒷판까지 이어진 기둥 → 서포트 불필요)
    bosses = [(xi0, xi0 + BOSS_W), (xi1 - BOSS_W, xi1)]
    mag_pts = []
    for bx0, bx1 in bosses:
        body += box(bx0 - 0.1, bx1, BACK - 0.1, yd, zi1 - BOSS_H, zi1 + 0.1)
        mx, mz = (bx0 + bx1) / 2, zi1 - BOSS_H / 2
        mag_pts.append((mx, mz))
        body -= cyl_y(yd - MAG_T, yd + 0.1, mx, mz, MAG_D / 2)

    # 너클 아래 45° 보강 (출력 시 오버행 지지)
    body += prism_x(xi0 - 0.1, xi1 + 0.1,
                    [(yd - 3, zi0 - 0.1), (yd, zi0 - 0.1), (yd, zi0 + 3)])

    # 본체 너클
    for i, (x0, x1) in enumerate(segs):
        if i % 2 == 0:
            a = x0 + (KNUCKLE_GAP / 2 if i > 0 else 0)
            b = x1 - (KNUCKLE_GAP / 2 if i < KNUCKLE_N - 1 else 0)
            body += cyl_x(a, b, axis_y, axis_z, KNUCKLE_R)
    # 문 너클 회전 공간 확보
    for i, (x0, x1) in enumerate(segs):
        if i % 2 == 1:
            body -= cyl_x(x0 - KNUCKLE_GAP / 2, x1 + KNUCKLE_GAP / 2, axis_y, axis_z, KNUCKLE_CUT)
    body -= cyl_x(-1, W + 1, axis_y, axis_z, PIN_D / 2)

    # 고정용 접시머리 나사 구멍 (섹터1 뒷판 상단)
    if SCREW_HOLES:
        sz = zi1 - BOSS_H - 12
        for sx in (xi0 + 22, xi0 + s1_w - 22):
            body -= cyl_y(-1, BACK + 1, sx, sz, SCREW_D / 2)
            body -= cyl_y(BACK - (SCREW_HEAD_D - SCREW_D) / 2 - 0.01, BACK + 0.01, sx, sz,
                          SCREW_D / 2, SCREW_HEAD_D / 2)

    # ------------------------------ 문 ------------------------------------
    dx0, dx1 = xi0 + GAP, xi1 - GAP
    dz0, dz1 = zi0 + GAP, zi1 - GAP
    door = box(dx0, dx1, yd, D, dz0, dz1)
    for i, (x0, x1) in enumerate(segs):
        if i % 2 == 1:
            door += cyl_x(x0 + KNUCKLE_GAP / 2, x1 - KNUCKLE_GAP / 2, axis_y, axis_z, KNUCKLE_R)
    for i, (x0, x1) in enumerate(segs):
        if i % 2 == 0:
            door -= cyl_x(x0 - KNUCKLE_GAP / 2, x1 + KNUCKLE_GAP / 2, axis_y, axis_z, KNUCKLE_CUT)
    door -= cyl_x(-1, W + 1, axis_y, axis_z, PIN_D / 2)
    for mx, mz in mag_pts:
        door -= cyl_y(yd - 0.1, yd + MAG_T, mx, mz, MAG_D / 2)

    # 독수리 부조
    span = min(100.0, (dx1 - dx0) * 0.70)
    xs, zs, Hm = eagle_heightmap(span=span)
    eagle = heightfield_solid(xs, zs, Hm)
    ez = (dz0 + dz1) / 2 - 10.0 * span / 128.0 + 2.0   # 설계 z(-31~52)의 중앙을 문 중앙에
    eagle = eagle.translate([W / 2, D, ez])
    door = door + eagle

    info = dict(preset=preset, W=W, D=D, H=Ht, s1_w=s1_w, s2_w=s2_w, in_d=in_d, in_h=in_h,
                knuckle_front=D + KNUCKLE_R, axis=(axis_y, axis_z),
                door=(dx1 - dx0, dz1 - dz0), mag_pts=mag_pts)
    return body, door, info


def door_rotated(door, info, deg):
    """힌지 축 기준으로 문을 회전 (양수 = 열림)"""
    ay, az = info["axis"]
    return door.translate([0, -ay, -az]).rotate([-deg, 0, 0]).translate([0, ay, az])


def save_stl(man, path):
    import trimesh
    mesh = man.to_mesh()
    tm = trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=False)
    tm.export(path)
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="regular", choices=PRESETS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "stl"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    body, door, info = build(args.preset)
    # 출력 방향: 뒷면이 베드(z=0)
    body_print = body.rotate([-90, 0, 0]).rotate([180, 0, 0])
    b = body_print.bounding_box()
    body_print = body_print.translate([-b[0], -b[1], -b[2]])
    yd = info["D"] - DOOR_T
    door_print = door.translate([0, -yd, 0]).rotate([-90, 0, 0]).rotate([180, 0, 0])
    b = door_print.bounding_box()
    door_print = door_print.translate([-b[0], -b[1], -b[2]])

    tb = save_stl(body_print, os.path.join(args.out, f"body_{args.preset}.stl"))
    td = save_stl(door_print, os.path.join(args.out, f"door_{args.preset}.stl"))
    # 미리보기용 조립 상태 (닫힘 / 열림)
    save_stl(body + door, os.path.join(args.out, f"_assembly_closed_{args.preset}.stl"))
    save_stl(body + door_rotated(door, info, 100), os.path.join(args.out, f"_assembly_open_{args.preset}.stl"))

    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()})
    for name, tm in (("body", tb), ("door", td)):
        print(name, "watertight=", tm.is_watertight, "volume_cm3=", round(tm.volume / 1000, 1),
              "bbox=", np.round(tm.extents, 1), "tris=", len(tm.faces))


if __name__ == "__main__":
    main()
