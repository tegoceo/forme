"""측단면 설명도 생성: python3 draw.py → preview/section.png"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import model as M

try:
    import koreanize_matplotlib  # noqa: F401  (NanumGothic)
except ImportError:
    pass
L, body, lid, eagle, eagle_on = M.build_all()
p, D, W = L["P"], L["D"], L["W"]
packs, lighter = M.items_solids(L)


def section(man, x0):
    cs = man.rotate([0, 90, 0]).slice(-x0)
    return [[(y, z) for z, y in poly] for poly in cs.to_polygons()]


def draw_polys(ax, polys, **kw):
    """외곽/구멍 윤곽을 하나의 경로로 그려 구멍(홈)이 비어 보이게 함"""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    verts, codes = [], []
    for poly in polys:
        verts += list(poly) + [poly[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY]
    if verts:
        ax.add_patch(PathPatch(Path(verts, codes), **kw))


x_mid = L["lx0"] + M.TC + 30          # 담배 칸 중앙
x_wall = M.WALL - 1.0                 # 측벽 홈 깊이 안
x_cheek = L["lx0"] + 1.0

fig, axes = plt.subplots(1, 3, figsize=(17, 7.6), dpi=150, gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
for ax, deg, title in ((axes[0], 0.0, "닫힘 (0°)"), (axes[1], M.THETA_MAX, f"최대 개방 ({M.THETA_MAX:.0f}°)")):
    ax.add_patch(Rectangle((-14, -6), 14, L["H"] + 12, color="#e4e1dc", zorder=0))
    ax.set_facecolor("#ffffff")
    ax.text(-7, L["H"] / 2, "벽", ha="center", va="center", color="#777")
    # 함체: 측벽(홈 포함) 단면을 옅게, 중앙 단면을 진하게
    draw_polys(ax, section(body, x_wall), facecolor="#d3cbbd", edgecolor="#8f8676", lw=0.7, zorder=1)
    draw_polys(ax, section(body, x_mid), facecolor="#bdb6aa", edgecolor="#6d665c", lw=0.8, zorder=2)
    lr = M.rot3d(lid + eagle_on, p, deg)
    draw_polys(ax, section(lr, x_cheek), facecolor="#9fb6d3", edgecolor="#3d5a80", lw=0.8, alpha=0.55, zorder=3)
    draw_polys(ax, section(lr, x_mid), facecolor="#5f86b8", edgecolor="#2b4a72", lw=0.8, zorder=4)
    draw_polys(ax, section(M.rot3d(packs[0], p, deg), x_mid), facecolor="#c8553d", edgecolor="#7a2c1d",
               lw=0.8, alpha=0.85, zorder=5)
    # 보스
    for (y, z), name in ((p, "피벗 P"), (L["G"], "가이드 G")):
        a = math.radians(-deg)
        yy = p[0] + (y - p[0]) * math.cos(a) - (z - p[1]) * math.sin(a)
        zz = p[1] + (y - p[0]) * math.sin(a) + (z - p[1]) * math.cos(a)
        ax.add_patch(Circle((yy, zz), M.BOSS_R, facecolor="#f2c14e", edgecolor="#8a6d1e", lw=1, zorder=6))
        if deg == 0:
            ax.annotate(name, (yy, zz), (yy - 30 if name.startswith("가") else yy + 12, zz - 16),
                        arrowprops=dict(arrowstyle="-", color="#8a6d1e", lw=0.8), fontsize=9, color="#6b5416")
    # 래치
    yl1 = D - M.T_F - M.LATCH_FRONT_GAP
    ax.add_patch(Rectangle((yl1 - M.LATCH_L, L["z_tu"] - M.LATCH_H), M.LATCH_L, M.LATCH_H, fill=False,
                           ls="--", ec="#2a9d8f", lw=1.4, zorder=7))
    ax.text(yl1 - M.LATCH_L / 2, L["z_tu"] - M.LATCH_H / 2, "푸시래치\n공간", ha="center", va="center",
            fontsize=8.5, color="#1d6f65", zorder=8)
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-14, D + 75)
    ax.set_ylim(-8, L["H"] + 10)
    ax.set_xlabel("깊이 y (mm, 0 = 벽면)")
    ax.grid(alpha=0.25)

ax = axes[0]
ax.set_ylabel("높이 z (mm)")
ax.annotate("", (0, L["H"] + 4), (D, L["H"] + 4), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.text(D / 2, L["H"] + 5.5, f"{D:.0f}", ha="center", fontsize=9)
ax.annotate("", (D + 8, 0), (D + 8, L["H"]), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.text(D + 10, L["H"] / 2, f"{L['H']:.1f}", va="center", fontsize=9)
ax.text(D + 14, 40, "흰색 = 측벽 안쪽 홈 (깊이 2 mm)\n · 수평 진입홈: 조립 시 보스가 들어가는 길\n · 원호 가이드홈: 열림 경로 + 35° 스토퍼", fontsize=8.5, color="#4a4438")
gy = p[0] + M.R_G * math.cos(math.radians(M.PHI_G - 20)); gz = p[1] + M.R_G * math.sin(math.radians(M.PHI_G - 20))
ax.annotate("", (gy, gz + 3.5), (D + 13, 41), arrowprops=dict(arrowstyle="->", color="#4a4438", lw=0.8))
ax = axes[1]
a = math.radians(M.THETA_MAX)
ax.plot([p[0], p[0]], [p[1], p[1] + 75], color="#555", lw=0.7, ls=":")
ax.plot([p[0], p[0] + 75 * math.sin(a)], [p[1], p[1] + 75 * math.cos(a)], color="#555", lw=0.7, ls=":")
ax.text(p[0] + 12, p[1] + 60, f"{M.THETA_MAX:.0f}°", fontsize=11)
ax = axes[2]
draw_polys(ax, section(body, x_wall), facecolor="#d3cbbd", edgecolor="#8f8676", lw=0.7)
def boss_at(pt, deg):
    a = math.radians(-deg)
    return (p[0] + (pt[0] - p[0]) * math.cos(a) - (pt[1] - p[1]) * math.sin(a),
            p[1] + (pt[0] - p[0]) * math.sin(a) + (pt[1] - p[1]) * math.cos(a))
for deg, col, lab in ((M.THETA_ENTRY, "#9aa5b1", f"조립 진입 {M.THETA_ENTRY:.0f}°"), (0, "#f2c14e", "닫힘 0°"),
                      (M.THETA_MAX, "#e76f51", f"최대 {M.THETA_MAX:.0f}°")):
    for pt in (p, L["G"]):
        c = boss_at(pt, deg)
        ax.add_patch(Circle(c, M.BOSS_R, facecolor=col, edgecolor="#333", lw=0.6, alpha=0.9))
    c = boss_at(L["G"], deg)
    ax.annotate(lab, c, (c[0] - 18, c[1] + (14 if deg == M.THETA_MAX else -10 if deg == 0 else -20)),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.text(p[0] + 4, p[1] - 9, "피벗 P (고정 회전축)", fontsize=8.5)
ax.annotate("앞쪽으로 열린 진입홈\n(래치 설치 후엔 -5°까지\n눌리지 않아 빠지지 않음)", (D - 1, p[1]), (D - 34, p[1] + 34), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.set_xlim(5, D + 5); ax.set_ylim(-3, 60); ax.set_aspect("equal"); ax.grid(alpha=0.25)
ax.set_title("측벽 안쪽 홈과 보스 위치 (확대)", fontsize=12)
ax.set_xlabel("깊이 y (mm)")
fig.suptitle("측단면 (담배 칸 중앙)  ·  회색 = 함체, 파랑 = 뚜껑, 빨강 = 담배갑, 노랑 = 결합 보스", fontsize=11)
fig.tight_layout()
fig.savefig("preview/section.png")
print("saved")
