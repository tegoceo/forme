"""독수리 부조 높이맵 생성 (model.py 에서 사용)"""

import math

import numpy as np
import manifold3d as m3d
from scipy.ndimage import gaussian_filter


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


