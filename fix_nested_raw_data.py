import shutil
from pathlib import Path

root = Path(r"C:\Users\Admin\Downloads\3D_Que\Raw_data")
required_files = {"camera_matrix.csv", "odometry.csv", "rgb.mp4"}
required_dirs = {"depth", "confidence"}

def has_required(p: Path) -> bool:
    return all((p / f).is_file() for f in required_files) and all((p / d).is_dir() for d in required_dirs)

def find_candidate(parent: Path, max_depth: int = 3):
    # BFS over subdirectories (skip __MACOSX)
    level = [c for c in parent.iterdir() if c.is_dir() and c.name != "__MACOSX"]
    depth = 1
    while level and depth <= max_depth:
        next_level = []
        for d in level:
            if has_required(d):
                return d
            next_level.extend([c for c in d.iterdir() if c.is_dir() and c.name != "__MACOSX"])
        level = next_level
        depth += 1
    return None

fixed, skipped = [], []

for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
    if has_required(folder):
        # clean junk if present
        mac = folder / "__MACOSX"
        if mac.exists() and mac.is_dir():
            shutil.rmtree(mac, ignore_errors=True)
        continue

    candidate = find_candidate(folder, max_depth=3)
    if candidate is None:
        skipped.append((folder.name, "khong_tim_thay_data_goc"))
        continue

    conflicts = []
    moved = []
    for item in candidate.iterdir():
        dest = folder / item.name
        if dest.exists():
            conflicts.append(item.name)
            continue
        shutil.move(str(item), str(dest))
        moved.append(item.name)

    # remove common junk
    mac = folder / "__MACOSX"
    if mac.exists() and mac.is_dir():
        shutil.rmtree(mac, ignore_errors=True)

    # try to clean empty wrappers up from candidate
    cur = candidate
    for _ in range(4):
        try:
            if cur.exists() and cur.is_dir() and not any(cur.iterdir()):
                cur.rmdir()
                cur = cur.parent
            else:
                break
        except Exception:
            break

    if moved and not conflicts:
        fixed.append((folder.name, f"moved:{len(moved)}"))
    elif moved and conflicts:
        fixed.append((folder.name, f"moved:{len(moved)}, conflict:{','.join(conflicts)}"))
    else:
        skipped.append((folder.name, "khong_move_duoc"))

print("FIXED_COUNT", len(fixed))
for n, m in fixed:
    print("FIXED", n, m)
print("SKIPPED_COUNT", len(skipped))
for n, m in skipped:
    print("SKIPPED", n, m)
