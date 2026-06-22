import prior
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, Point



def world_to_pix(x, z, xmin, zmin, ppm):
    col = int(round((x - xmin) * ppm))
    row = int(round((z - zmin) * ppm))
    return col, row

def polygon_to_mask(poly_xz, xmin, zmin, ppm, H, W):
    poly = Polygon(poly_xz)
    # Sample grid centers
    ys, xs = np.mgrid[0:H, 0:W]
    xs = xs + 0.5
    ys = ys + 0.5
    # back to world:
    wx = (xs / ppm) + xmin
    wz = (ys / ppm) + zmin
    # Vectorized contains check in batches
    mask = np.zeros((H, W), dtype=bool)
    # process in tiles to limit Python overhead
    tile = 512
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            yy = slice(y0, min(H, y0 + tile))
            xx = slice(x0, min(W, x0 + tile))
            coords = np.dstack([wx[yy, xx], wz[yy, xx]]).reshape(-1, 2)
            mask_block = np.array([poly.contains(Point(c[0], c[1])) or poly.touches(Point(c[0], c[1])) for c in coords])
            mask[yy, xx] = mask_block.reshape((yy.stop - yy.start, xx.stop - xx.start))
    return mask

def cc_label(binary):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(h):
        for x in range(w):
            if not binary[y, x]:
                continue
            neighbors = []
            if y > 0 and labels[y - 1, x] > 0: neighbors.append(labels[y - 1, x])
            if x > 0 and labels[y, x - 1] > 0: neighbors.append(labels[y, x - 1])
            if not neighbors:
                current += 1
                parent.append(current)
                labels[y, x] = current
            else:
                m = min(neighbors)
                labels[y, x] = m
                for n in neighbors:
                    if n != m:
                        union(m, n)
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                labels[y, x] = find(labels[y, x])
    uniq = np.unique(labels[labels > 0])
    remap = {u: i + 1 for i, u in enumerate(uniq)}
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                labels[y, x] = remap[labels[y, x]]
    return labels

def thinning_zhang_suen(binary):
    img = binary.astype(np.uint8).copy()
    changing1 = changing2 = [1]
    h, w = img.shape
    while changing1 or changing2:
        changing1 = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = img[y-1,x],img[y-1,x+1],img[y,x+1],img[y+1,x+1],img[y+1,x],img[y+1,x-1],img[y,x-1],img[y-1,x-1]
                A = (P2==0 and P3==1)+(P3==0 and P4==1)+(P4==0 and P5==1)+(P5==0 and P6==1)+ \
                    (P6==0 and P7==1)+(P7==0 and P8==1)+(P8==0 and P9==1)+(P9==0 and P2==1)
                B = P2+P3+P4+P5+P6+P7+P8+P9
                if img[y,x]==1 and 2<=B<=6 and A==1 and (P2*P4*P6==0) and (P4*P6*P8==0):
                    changing1.append((y,x))
        for y,x in changing1: img[y,x]=0

        changing2 = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = img[y-1,x],img[y-1,x+1],img[y,x+1],img[y+1,x+1],img[y+1,x],img[y+1,x-1],img[y,x-1],img[y-1,x-1]
                A = (P2==0 and P3==1)+(P3==0 and P4==1)+(P4==0 and P5==1)+(P5==0 and P6==1)+ \
                    (P6==0 and P7==1)+(P7==0 and P8==1)+(P8==0 and P9==1)+(P9==0 and P2==1)
                B = P2+P3+P4+P5+P6+P7+P8+P9
                if img[y,x]==1 and 2<=B<=6 and A==1 and (P2*P4*P8==0) and (P2*P6*P8==0):
                    changing2.append((y,x))
        for y,x in changing2: img[y,x]=0
    return img.astype(bool)

def door_points_from_metadata(house, xmin, zmin, ppm):
    pts = []

    if 'doors' in house and isinstance(house['doors'], list):
        for d in house['doors']:
            print(d)
            print('\n')
            if d["openable"]:
                x, z = d["assetPosition"]["x"], d["assetPosition"]["y"]
                c, r = world_to_pix(x, z, xmin, zmin, ppm)
                pts.append((r, c))

    uniq = {}
    for (r,c) in pts:
        uniq[(int(r), int(c))] = True
    return list(uniq.keys())

def parse_wall_str(wall_str):
    # "wall|<room_idx>|x0|z0|x1|z1"
    parts = wall_str.split("|")
    if len(parts) != 6 or parts[0] != "wall":
        raise ValueError(f"Unexpected wall format: {wall_str}")
    _, room_idx, x0, z0, x1, z1 = parts
    return float(x0), float(z0), float(x1), float(z1)

def door_segment_from_metadata(door):
    """
    Returns: (wx0, wz0, wx1, wz1) world coords for the short cross-bar
    centered on the doorway, perpendicular to the wall.
    """
    # Use the first wall; both walls describe the same opening from either room
    x0, z0, x1, z1 = parse_wall_str(door["wall0"])
    hole = door["holePolygon"]
    # distances along the wall from its start (meters)
    d0 = float(hole[0]["x"])
    d1 = float(hole[1]["x"])

    # wall direction (unit) and its perpendicular (unit)
    wx, wz = x1 - x0, z1 - z0
    L = (wx**2 + wz**2) ** 0.5
    if L == 0:
        return None
    ux, uz = wx / L, wz / L                       # along the wall
    nx, nz = -uz, ux                               # perpendicular to wall (outward normal)

    # world endpoints of the opening interval along the wall
    a_x, a_z = x0 + ux * d0, z0 + uz * d0
    b_x, b_z = x0 + ux * d1, z0 + uz * d1
    # midpoint of the opening
    m_x, m_z = (a_x + b_x) * 0.5, (a_z + b_z) * 0.5

    # pick a short length across the doorway (e.g., 0.15 m each side = 0.30 m total)
    half_len_m = 0.15
    # half_len_m = 0.5 * (d1 - d0)
    p0_x, p0_z = m_x - nx * half_len_m, m_z - nz * half_len_m
    p1_x, p1_z = m_x + nx * half_len_m, m_z + nz * half_len_m
    return p0_x, p0_z, p1_x, p1_z

def world_to_pix(x, z, xmin, zmin, ppm):
    # same mapping you already use elsewhere
    col = int(round((x - xmin) * ppm))
    row = int(round((z - zmin) * ppm))
    return col, row

def draw_doors_from_house(house, ax, xmin, zmin, ppm, flip_top_bottom, H):
    """
    Draw short cross-bars for all doors in the house onto axes 'ax'.
    """
    if "doors" not in house:
        return
    # y-flip mapper (if you’re still using --flip_tb)
    fy = (lambda y: H - 1 - y) if flip_top_bottom else (lambda y: y)

    for d in house["doors"]:
        try: d["openable"]
        except: continue
        if d["openable"]:
            seg = door_segment_from_metadata(d)
            if seg is None:
                continue
            p0x, p0z, p1x, p1z = seg
            # world -> pixel
            x0, y0 = world_to_pix(p0x, p0z, xmin, zmin, ppm)
            x1, y1 = world_to_pix(p1x, p1z, xmin, zmin, ppm)
            # draw
            ax.plot([x0, x1], [fy(y0), fy(y1)],
                    linewidth=20, color="blue")
        
def door_points_geometric_fallback(room_label_img, floor_mask, narrow_radius_px=3, min_cluster=3):
    skel = skeletonize(floor_mask)
    dist = distance_transform_edt(floor_mask)

    narrow = (dist <= narrow_radius_px)
    door_cand = skel & narrow

    # Needs at least 2 distinct room IDs in a neighborhood
    H, W = room_label_img.shape
    door_mask = np.zeros_like(door_cand, dtype=bool)
    ys, xs = np.where(door_cand)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-3), min(H, y+4)
        x0, x1 = max(0, x-3), min(W, x+4)
        vals = np.unique(room_label_img[y0:y1, x0:x1])
        vals = vals[vals > 0]
        if vals.size >= 2:
            door_mask[y, x] = True

    labs = sk_label(door_mask, connectivity=1)
    pts = []
    for r in regionprops(labs):
        if r.area >= min_cluster:
            y, x = r.centroid
            pts.append((int(y), int(x)))
    return pts

def draw_robot(ax, pose, xmin, zmin, ppm, H, flip_top_bottom=False,
               base_m=0.40, height_m=0.55, back_m=0.00,
               facecolor="tab:red", edgecolor="black", alpha=0.9, zorder=10):
    """
    Draw a triangle for the robot at world pose (x,y,z,yaw).

    Args
      ax: matplotlib axes
      pose: dict like {"x":..., "y":..., "z":..., "yaw":...} (yaw in degrees).
            Convention: yaw=0 faces +Z, yaw=90 faces +X (AI2-THOR default).
      xmin, zmin, ppm, H: same rasterization params you already use
      flip_top_bottom: match your --flip_tb rendering
      base_m: triangle base width in meters
      height_m: tip distance forward from robot position in meters
      back_m: shift base backward along -forward (meters), usually 0
    """
    xw, zw = float(pose["x"]), float(pose["z"])
    yaw_deg = float(pose["yaw"])
    th = np.deg2rad(yaw_deg)

    # Forward (+Z at 0°, +X at 90°)
    fx, fz = np.sin(th), np.cos(th)
    # Right-hand perpendicular (for base width)
    rx, rz = fz, -fx

    # Triangle vertices in world (meters)
    tip_x,  tip_z  = xw + fx * height_m,          zw + fz * height_m
    base_mid_x, base_mid_z = xw - fx * back_m,    zw - fz * back_m
    left_x, left_z   = base_mid_x - rx * (base_m/2), base_mid_z - rz * (base_m/2)
    right_x, right_z = base_mid_x + rx * (base_m/2), base_mid_z + rz * (base_m/2)

    # World → pixel
    def world_to_pix(x, z, xmin, zmin, ppm):
        col = int(round((x - xmin) * ppm))
        row = int(round((z - zmin) * ppm))
        return col, row

    tx, ty = world_to_pix(tip_x,   tip_z,   xmin, zmin, ppm)
    lx, ly = world_to_pix(left_x,  left_z,  xmin, zmin, ppm)
    rxp, ry = world_to_pix(right_x, right_z, xmin, zmin, ppm)

    # Optional vertical flip to match your map orientation
    fy = (lambda y: H - 1 - y) if flip_top_bottom else (lambda y: y)

    ax.fill([lx, rxp, tx], [fy(ly), fy(ry), fy(ty)],
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=1.5, zorder=zorder)

    # (Optional) draw a tiny heading tick from base_mid to emphasize direction:
    # mx, my = world_to_pix(base_mid_x, base_mid_z, xmin, zmin, ppm)
    # hx, hy = world_to_pix(base_mid_x + fx*0.15, base_mid_z + fz*0.15, xmin, zmin, ppm)
    # ax.plot([mx, hx], [fy(my), fy(hy)], color=edgecolor, linewidth=1.2, zorder=zorder+1)

def build_skeleton_and_doors(house, agent_pos, ppm=80, pad_m=0.5, out_path="out.png"):
    rooms = house.get("rooms", [])
    if not rooms:
        raise ValueError("No 'rooms' in house JSON.")

    polys = []
    for r in rooms:
        pts = [(v["x"], v["z"]) for v in r["floorPolygon"]]
        polys.append((r["id"], r["roomType"], pts))

    xs = [p[0] for _,_,pts in polys for p in pts]
    zs = [p[1] for _,_,pts in polys for p in pts]
    xmin, xmax = min(xs)-pad_m, max(xs)+pad_m
    zmin, zmax = min(zs)-pad_m, max(zs)+pad_m

    W = int(math.ceil((xmax - xmin) * ppm))
    H = int(math.ceil((zmax - zmin) * ppm))

    room_id_to_label = {}
    next_label = 1
    label_img = np.zeros((H, W), dtype=np.int32)
    for room_id, room_type, pts in polys:
        room_name = room_id + " - " + room_type
        mask = polygon_to_mask(pts, xmin, zmin, ppm, H, W)
        if room_name not in room_id_to_label:
            room_id_to_label[room_name] = next_label
            next_label += 1
        lab = room_id_to_label[room_name]
        label_img[mask] = lab

    floor_mask = label_img > 0
    floor_mask = remove_small_objects(floor_mask, 50)

    base = np.ones((H, W, 3), dtype=float)
    base *= 1.0

    rng = np.random.default_rng(0)
    palette = {}
    # room_id_to_label was created earlier
    for rid, lab in room_id_to_label.items():
        palette[lab] = rng.uniform(0.85, 0.98, size=3)
    for lab in np.unique(label_img):
        if lab == 0:
            continue
        base[label_img == lab] = palette.get(lab, [0.95, 0.95, 0.98])

    fy = (lambda y: H - 1 - y)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(base[::-1] if True else base, origin='upper')

    draw_doors_from_house(house, ax, xmin, zmin, ppm, flip_top_bottom=True, H=H)
    draw_robot(ax, agent_pos, xmin=xmin, zmin=zmin, ppm=ppm, H=H, flip_top_bottom=True)

    # room labels at region centroids (use raster centroids)
    for room_id, lab in room_id_to_label.items():
        yy, xx = np.where(label_img == lab)
        if len(xx) == 0:
            continue
        cx, cy = float(np.mean(xx)), float(np.mean(yy))
        ax.text(cx, fy(cy), f"{room_id}", fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.75))

    ax.set_title("ProcTHOR rooms (IDs), skeleton, doors")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close('all')
    # print(f"Saved: {out_path}")

def main():
    dataset = prior.load_dataset("procthor-10k")
    dataset = dataset["train"]
    house = dataset[20]
    agent_pose = {"x":3.75, "y":0.9, "z":6.0, "yaw":90}

    build_skeleton_and_doors(house, agent_pose, ppm=40, out_path="skeleton_fp.png")

if __name__ == "__main__":
    main()
