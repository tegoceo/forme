"""조립/동작 간섭 검증: python3 check.py"""
import math
import numpy as np
import model as M

L, body, lid, eagle, eagle_on = M.build_all()
p = L["P"]
lid_full = lid + eagle_on
packs, lighter = M.items_solids(L)
items = packs[0] + packs[1] + lighter

def inter(a, b):
    return (a ^ b).volume()

def lid_at(deg, dy=0.0, dz=0.0):
    return M.rot3d(lid_full, p, deg).translate([0, dy, dz])

ok = True
def report(name, cond, detail):
    global ok
    ok &= cond
    print(f"[{'OK' if cond else 'NG'}] {name}: {detail}")

# 1. 작동 범위 전체 간섭 없음
vals = [(a, inter(body, lid_at(a))) for a in np.arange(M.THETA_ENTRY, M.THETA_MAX + 0.01, 1.0)]
worst = max(v for _, v in vals)
report("회전 간섭 (-5~35°, 1° 간격)", worst < 1e-3, f"최대 겹침 {worst:.4f} mm3")
# 2. 스토퍼
for a in (M.THETA_MAX + 1.5, M.THETA_ENTRY - 1.5):
    v = inter(body, lid_at(a))
    report(f"스토퍼 {a:+.1f}° 에서 걸림", v > 0.5, f"겹침 {v:.2f} mm3")
# 3. 빠짐 방지 (사용 범위에서 앞으로 당겨도 안 빠짐)
for a in (M.THETA_OVER, 0.0, 10.0, 20.0, M.THETA_MAX):
    v = inter(body, lid_at(a, dy=1.5))
    report(f"{a:+.0f}° 에서 앞으로 1.5mm 당김 → 걸림", v > 0.5, f"겹침 {v:.2f} mm3")
for a in (0.0, M.THETA_MAX):
    v = inter(body, lid_at(a, dz=1.5))
    report(f"{a:+.0f}° 에서 위로 1.5mm 들어올림 → 걸림", v > 0.5, f"겹침 {v:.2f} mm3")
# 4. 조립 경로 (진입각에서 앞으로 빼냄)
vs = [inter(body, lid_at(M.THETA_ENTRY, dy=t)) for t in np.arange(0, L["D"] + 5, 2.0)]
report("조립 경로 (진입각 -5° 에서 직선 삽입)", max(vs) < 1e-3, f"최대 겹침 {max(vs):.4f} mm3")
# 5. 내용물 회전 간섭
vs = [inter(body, M.rot3d(items, p, a)) for a in np.arange(M.THETA_OVER, M.THETA_MAX + 0.01, 1.0)]
report("담배갑/라이터 회전 간섭 (-2~35°)", max(vs) < 1e-3, f"최대 겹침 {max(vs):.4f} mm3")
v = inter(lid, items)
report("내용물이 뚜껑 칸 안에 들어감", v < 1e-3, f"겹침 {v:.4f} mm3")
# 6. 래치 몸체 공간 (래치 박스 vs 뚜껑/내용물, 사용 범위)
xc, D = L["xc"], L["D"]
yl1 = D - M.T_F - M.LATCH_FRONT_GAP
latch = M.box(xc - M.LATCH_W / 2, xc + M.LATCH_W / 2, yl1 - M.LATCH_L, yl1, L["z_tu"] - M.LATCH_H, L["z_tu"])
v1 = inter(latch, body)
vs = [inter(latch, lid_at(a)) + inter(latch, M.rot3d(items, p, a)) for a in np.arange(M.THETA_OVER, M.THETA_MAX + 0.01, 1.0)]
report("래치 설치 공간 비어 있음", v1 < 1e-3, f"함체와 겹침 {v1:.4f} mm3")
report("래치 vs 뚜껑/내용물 (-2~35°)", max(vs) < 1e-3, f"최대 겹침 {max(vs):.4f} mm3")
v = inter(latch, lid_at(M.THETA_ENTRY))
report("래치 설치 후 진입각 도달 불가 (= 뚜껑 안 빠짐)", v > 0.5, f"겹침 {v:.2f} mm3")

# 7. 꺼내기 각도 평가: 최대 개방 시 갑 뒷면이 함체 앞면(y=D) 밖으로 나온 길이
def exposed(deg, depth, h):
    a = math.radians(deg)
    yb = D - M.T_F - depth - p[0]
    out = []
    for s in np.linspace(0, h, 200):
        dz = L["z_floor"] + h - s - p[1]
        y = p[0] + yb * math.cos(a) + dz * math.sin(a)
        if y >= D:
            out.append(s)
    return max(out) if out else 0.0
for a in (20, 25, 30, M.THETA_MAX):
    print(f"   개방 {a:>4.0f}°: 담배갑 뒷면 노출 {exposed(a, M.PACK_D, M.PACK_H):5.1f} mm / "
          f"라이터 {exposed(a, M.LIGHTER_D, M.LIGHTER_H):5.1f} mm  "
          f"(앞판 상단 돌출 {(L['z_pt'] - p[1]) * math.sin(math.radians(a)):.0f} mm)")
print("ALL OK" if ok else "CHECK FAILED")
