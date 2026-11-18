"""벽부형 담배 케이스 v2 — 틸트아웃(앞으로 기울어지며 열리는) 구조

구성
  body  : 함체. 벽에 양면테이프로 부착. 상단에 푸시래치 설치 공간.
  lid   : 뚜껑(빈). 앞판 + 좌우 측면판 + 바닥 + 칸막이 + 뒤턱.
          담배갑 2개와 라이터가 앞판 안쪽에 기대어 있다가 뚜껑과 함께 기울어져 나옴.
  eagle : 독수리 부조. 별도 출력 후 앞판에 접착 (1.75 필라멘트 토막 2개로 위치 맞춤).

결합 구조 (외부에서 보이지 않음)
  - 측면판 바깥쪽 보스 2개(피벗 P, 가이드 G) → 함체 측벽 안쪽 홈에 끼워짐.
  - 피벗 P 를 중심으로 회전. 가이드 G 는 원호 홈을 따라 움직여 경로/각도를 제한.
  - 원호 홈 끝 = 최대 개방각(THETA_MAX), 반대쪽 끝 = 조립 진입각(THETA_ENTRY).
  - 조립: 푸시래치를 달기 전에 뚜껑을 THETA_ENTRY 각도로 앞에서 밀어 넣고 닫는다.
    래치를 설치하면 뚜껑이 진입각까지 눌리지 않으므로 빠지지 않는다.

좌표계 (mm): x 폭, y 깊이(0 = 벽면, + = 앞), z 높이.
각도: 양수 = 열림(앞판 위쪽이 앞으로), 음수 = 안쪽으로 눌림.

사용법:  python3 model.py
"""

import math
import os
import sys

import numpy as np
import manifold3d as m3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle import eagle_heightmap, heightfield_solid  # noqa: E402

# ---------------------------------------------------------------------------
# 파라미터
# ---------------------------------------------------------------------------
PACK_W, PACK_D, PACK_H = 55.0, 23.0, 87.0     # 실측 담배갑
LIGHTER_W, LIGHTER_D, LIGHTER_H = 25.0, 15.0, 82.0

# 푸시래치 (몸체 기준, 제품 확정 후 수정)
LATCH_L, LATCH_W, LATCH_H = 39.0, 27.0, 15.0  # 깊이 x 폭 x 높이
LATCH_FRONT_GAP = 4.0                          # 래치 앞면 ~ 앞판 뒷면 (눌림 여유)

WALL = 4.0          # 함체 측벽 (홈 2.0 + 잔여 2.0)
BACK = 3.0          # 뒷판
TOP = 4.0           # 윗판 (래치 고정면)
BOT = 3.0           # 바닥판
T_F = 3.0           # 뚜껑 앞판
TC = 2.4            # 뚜껑 측면판
DIV = 2.0           # 칸막이
SIDE_GAP = 0.5      # 측면판 ~ 측벽
GAP = 0.5           # 회전 여유
TOP_GAP = 0.8       # 앞판 윗면 ~ 윗판

S1_W = 2 * PACK_W + 3.0      # 담배 칸 폭
S2_W = LIGHTER_W + 3.0       # 라이터 칸 폭
POCKET_D = PACK_D + 2.0      # 칸 깊이 (앞판 뒷면 ~ 뒤턱)
LIP_T, LIP_H = 2.0, 12.0     # 뒤턱 (닫힌 상태에서 갑이 뒤로 넘어가지 않게)

PIV_BACK = 5.0      # 피벗이 앞판 뒷면보다 뒤쪽으로 들어간 거리
Z_P = 13.0          # 피벗 높이
R_C = 86.0          # 측면판 상단 원호 반경 (피벗 중심)
R_PB = 12.0         # 앞판 하단 곡면 반경 (피벗 중심)
CHEEK_BELOW = 5.5   # 피벗 아래 측면판 여유

BOSS_R = 3.0        # 보스 Ø6
GROOVE_R = 3.15     # 홈 폭 6.3
GROOVE_DEPTH = 2.0
BOSS_LEN = SIDE_GAP + GROOVE_DEPTH - 0.2
R_G = 17.0          # 가이드 보스 반경

THETA_MAX = 35.0    # 최대 개방각
THETA_OVER = -2.0   # 래치 누름(오버트래블)
THETA_ENTRY = -5.0  # 조립 진입각 (래치 미설치 시에만 도달)
PHI_G = 180.0 + THETA_ENTRY   # 닫힘 상태에서 가이드 보스 방향 (피벗 기준, +y 에서 반시계)

HEADER_CLEAR = 1.0
EAGLE_SPAN = 100.0
PEG_D = 2.0         # 1.75 필라멘트 정렬핀 구멍

CS_SEG = 128


# ---------------------------------------------------------------------------
# 도형 헬퍼
# ---------------------------------------------------------------------------
def rect(y0, y1, z0, z1):
    return m3d.CrossSection.square([y1 - y0, z1 - z0]).translate([y0, z0])


def circle(cy, cz, r):
    return m3d.CrossSection.circle(r, CS_SEG).translate([cy, cz])


def union2d(items):
    return m3d.CrossSection.batch_boolean(list(items), m3d.OpType.Add)


def rot2d(cs, p, deg_open):
    """피벗 p 기준 회전. deg_open>0 이면 열림 방향(시계방향)."""
    return cs.translate([-p[0], -p[1]]).rotate(-deg_open).translate([p[0], p[1]])


def extrude_x(cs, x0, x1):
    """(y,z) 단면을 x 방향으로 돌출"""
    return cs.extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_x(x0, x1, y, z, r, seg=96):
    return m3d.Manifold.cylinder(x1 - x0, r, r, seg).rotate([0, 90, 0]).translate([x0, y, z])


def cyl_y(y0, y1, x, z, r, r2=None, seg=96):
    r2 = r if r2 is None else r2
    return m3d.Manifold.cylinder(y1 - y0, r, r2, seg).rotate([-90, 0, 0]).translate([x, y0, z])


def rot3d(man, p, deg_open):
    """3D 회전 (x축, 피벗 p=(y,z)). deg_open>0 = 열림"""
    return man.translate([0, -p[0], -p[1]]).rotate([-deg_open, 0, 0]).translate([0, p[0], p[1]])


# ---------------------------------------------------------------------------
# 치수 계산
# ---------------------------------------------------------------------------
def layout():
    L = {}
    L["Wi"] = 2 * TC + 2 * SIDE_GAP + S1_W + DIV + S2_W
    L["W"] = L["Wi"] + 2 * WALL
    L["D"] = max(BACK + LATCH_L + LATCH_FRONT_GAP + T_F,
                 BACK + 8.0 + POCKET_D + LIP_T + T_F)
    D = L["D"]
    L["P"] = (D - T_F - PIV_BACK, Z_P)
    L["y_rear"] = D - T_F - POCKET_D - LIP_T
    L["z_floor"] = Z_P + 2.0
    L["lx0"] = WALL + SIDE_GAP
    L["lx1"] = L["W"] - WALL - SIDE_GAP

    # 담배갑/라이터 (앞판 뒷면에 기댄 상태) 의 회전 궤적 최고점 → 래치 공간 하한
    p = L["P"]
    yb = D - T_F
    corners = []
    for d, h in ((PACK_D, PACK_H), (LIGHTER_D, LIGHTER_H)):
        for y in (yb - d, yb):
            corners.append((y - p[0], L["z_floor"] + h - p[1]))
    r_items = max(math.hypot(*c) for c in corners)
    L["r_items"] = r_items
    z_hb = p[1] + max(r_items, R_C) + HEADER_CLEAR
    L["z_latch_bot"] = z_hb
    L["z_tu"] = z_hb + LATCH_H + 0.5          # 윗판 아랫면
    L["H"] = L["z_tu"] + TOP
    L["z_pt"] = L["z_tu"] - TOP_GAP           # 앞판 윗면
    L["z_latch_c"] = L["z_tu"] - LATCH_H / 2
    L["xc"] = L["W"] / 2
    a = math.radians(PHI_G)
    L["G"] = (p[0] + R_G * math.cos(a), p[1] + R_G * math.sin(a))   # 닫힘 상태 가이드 보스
    return L


# ---------------------------------------------------------------------------
# 뚜껑 단면
# ---------------------------------------------------------------------------
def lid_profiles(L):
    D, p = L["D"], L["P"]
    panel = rect(D - T_F, D, p[1] - R_PB, L["z_pt"]) ^ (
        circle(p[0], p[1], R_PB) + rect(D - T_F - 1, D + 1, p[1], L["z_pt"] + 1))
    cheek = rect(L["y_rear"], D - T_F + 0.01, p[1] - CHEEK_BELOW, p[1] + R_C + 1) ^ circle(p[0], p[1], R_C)
    floor = rect(L["y_rear"], D - T_F + 0.01, p[1] - 1.0, p[1] + 2.0)
    lip = rect(L["y_rear"], L["y_rear"] + LIP_T, p[1] + 1.99, p[1] + 2.0 + LIP_H)
    return dict(panel=panel, cheek=cheek, floor=floor, lip=lip)


# ---------------------------------------------------------------------------
# 함체
# ---------------------------------------------------------------------------
def build_body(L):
    W, D, H, p = L["W"], L["D"], L["H"], L["P"]
    prof = lid_profiles(L)
    lid2d = union2d(prof.values()).offset(GAP, m3d.JoinType.Round)

    sweep = [rot2d(lid2d, p, a) for a in np.arange(THETA_ENTRY, THETA_MAX + 0.01, 1.0)]
    # 조립 경로: 진입각에서 앞으로 빼내는 궤적
    ent = rot2d(lid2d, p, THETA_ENTRY)
    sweep += [ent.translate([t, 0]) for t in np.arange(0, D + 2, 1.5)]
    cav = union2d(sweep + [
        rect(BACK, D + 1, p[1] - CHEEK_BELOW - 1.0, L["z_tu"]),
        rect(BACK, p[0] - R_PB - GAP, BOT, L["z_tu"]),
    ])
    body = box(0, W, 0, D, 0, H) - extrude_x(cav, WALL, W - WALL)

    # 측벽 안쪽 홈: 수평 진입홈(피벗+가이드 공용) + 가이드 원호홈
    groove = [rect(L["G"][0] - 1.0, D + 1, p[1] - GROOVE_R, p[1] + GROOVE_R)]
    for th in np.arange(THETA_ENTRY, THETA_MAX + 0.01, 0.5):
        a = math.radians(PHI_G - th)
        groove.append(circle(p[0] + R_G * math.cos(a), p[1] + R_G * math.sin(a), GROOVE_R))
    groove = union2d(groove)
    body -= extrude_x(groove, WALL - GROOVE_DEPTH, WALL + 0.01)
    body -= extrude_x(groove, W - WALL - 0.01, W - WALL + GROOVE_DEPTH)

    # 래치 위치 가이드 리브 (윗판 아래, 래치 좌우)
    xc = L["xc"]
    yl1 = D - T_F - LATCH_FRONT_GAP
    for sx in (-1, 1):
        x0 = xc + sx * (LATCH_W / 2 + 0.4)
        body += box(min(x0, x0 + sx * 1.2), max(x0, x0 + sx * 1.2),
                    yl1 - LATCH_L, yl1 - 12, L["z_tu"] - 3.0, L["z_tu"] + 0.1)

    # 보조 고정 나사 구멍 (래치 높이, 앞에서 드라이버 접근 가능)
    for sx in (30.0, W - 30.0):
        z = L["z_latch_c"]
        body -= cyl_y(-1, BACK + 1, sx, z, 2.1)
        body -= cyl_y(BACK - 2.1, BACK + 0.01, sx, z, 2.1, 4.25)
    return body


# ---------------------------------------------------------------------------
# 뚜껑 + 독수리
# ---------------------------------------------------------------------------
def build_lid(L):
    D, p = L["D"], L["P"]
    lx0, lx1 = L["lx0"], L["lx1"]
    prof = lid_profiles(L)
    lid = extrude_x(prof["panel"], lx0, lx1)
    lid += extrude_x(prof["floor"], lx0, lx1)
    lid += extrude_x(prof["lip"], lx0, lx1)
    lid += extrude_x(prof["cheek"], lx0, lx0 + TC)
    lid += extrude_x(prof["cheek"], lx1 - TC, lx1)
    xd = lx0 + TC + S1_W
    lid += extrude_x(prof["cheek"], xd, xd + DIV)

    for (y, z) in (p, L["G"]):
        lid += cyl_x(lx0 - BOSS_LEN, lx0 + 0.5, y, z, BOSS_R)
        lid += cyl_x(lx1 - 0.5, lx1 + BOSS_LEN, y, z, BOSS_R)

    # 래치 스트라이크(철판) 자리 1 mm
    zc, xc = L["z_latch_c"], L["xc"]
    lid -= box(xc - 10, xc + 10, D - T_F - 0.01, D - T_F + 1.0, zc - 10, min(zc + 10, L["z_pt"] + 1))

    # 독수리 위치/정렬핀 구멍
    k = EAGLE_SPAN / 128.0
    ez = (p[1] - R_PB * 0.8 + L["z_pt"]) / 2 - 10.5 * k
    L["eagle_z"] = ez
    pegs = [(xc, ez - 2.3), (xc, ez + 3.9)]
    L["pegs"] = pegs
    for (x, z) in pegs:
        lid -= cyl_y(D - 2.0, D + 0.1, x, z, PEG_D / 2)
    return lid


def build_eagle(L):
    xs, zs, Hm = eagle_heightmap(span=EAGLE_SPAN)
    mask = Hm > 0.2
    Hp = np.where(mask, Hm + 0.6, -1.0)          # 0.6 mm 바탕 두께
    sol = heightfield_solid(xs, zs, Hp, base=-1.0)
    sol = sol ^ box(-200, 200, 0, 20, -200, 200)  # y>=0 만
    for (x, z) in L["pegs"]:
        sol -= cyl_y(-0.1, 2.0, x - L["xc"], z - L["eagle_z"], PEG_D / 2)
    return sol  # 로컬: 뒷면 y=0, 중심 (0, 0)


def items_solids(L):
    D, zf = L["D"], L["z_floor"]
    x = L["lx0"] + TC
    packs = [box(x + 1.0, x + 1.0 + PACK_W, D - T_F - PACK_D, D - T_F - 0.05, zf + 0.05, zf + PACK_H),
             box(x + 2.0 + PACK_W, x + 2.0 + 2 * PACK_W, D - T_F - PACK_D, D - T_F - 0.05, zf + 0.05, zf + PACK_H)]
    xl = x + S1_W + DIV + 1.5
    lighter = box(xl, xl + LIGHTER_W, D - T_F - LIGHTER_D, D - T_F - 0.05, zf + 0.05, zf + LIGHTER_H)
    return packs, lighter


def build_all():
    L = layout()
    body = build_body(L)
    lid = build_lid(L)
    eagle = build_eagle(L)
    eagle_on = eagle.translate([L["xc"], L["D"], L["eagle_z"]])
    return L, body, lid, eagle, eagle_on


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
def save_stl(man, path):
    import trimesh
    mesh = man.to_mesh()
    tm = trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=False)
    tm.export(path)
    return tm


def to_bed(man):
    b = man.bounding_box()
    return man.translate([-b[0], -b[1], -b[2]])


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")
    os.makedirs(out, exist_ok=True)
    L, body, lid, eagle, eagle_on = build_all()

    parts = {
        "body.stl": to_bed(body.rotate([90, 0, 0])),           # 뒷판이 베드
        "lid.stl": to_bed(lid.rotate([-90, 0, 0])),            # 앞판(앞면)이 베드
        "eagle.stl": to_bed(eagle.rotate([90, 0, 0])),         # 부조 뒷면이 베드
    }
    for name, man in parts.items():
        tm = save_stl(man, os.path.join(out, name))
        print(f"{name:10s} watertight={tm.is_watertight} vol={tm.volume / 1000:.1f}cm3 "
              f"bbox={np.round(tm.extents, 1)} tris={len(tm.faces)}")

    p = L["P"]
    packs, lighter = items_solids(L)
    items = packs[0] + packs[1] + lighter
    full_lid = lid + eagle_on
    save_stl(body, os.path.join(out, "_asm_body.stl"))
    for tag, deg in (("closed", 0.0), ("open", THETA_MAX)):
        save_stl(rot3d(full_lid, p, deg), os.path.join(out, f"_asm_{tag}_lid.stl"))
        save_stl(rot3d(items, p, deg), os.path.join(out, f"_asm_{tag}_items.stl"))
    keys = ("W", "H", "D", "P", "G", "z_floor", "z_latch_bot", "z_tu", "z_pt", "r_items", "y_rear")
    print({k: (tuple(round(v, 2) for v in L[k]) if isinstance(L[k], tuple) else round(L[k], 2)) for k in keys})


if __name__ == "__main__":
    main()
