"""Auto-patch gym-pybullet-drones to support the surveillance drone model.

Call ensure_surveillance_drone() before creating any aviary.
It copies the URDF to the package assets and adds the enum entry if missing.
Idempotent — safe to call multiple times.
"""

import os
import shutil
import pkg_resources


def ensure_surveillance_drone():
    """Install the surveillance drone URDF and enum entry into gym-pybullet-drones."""
    _install_urdf()
    _patch_enum()


def _install_urdf():
    """Copy surveillance.urdf into the gym-pybullet-drones assets folder."""
    assets_dir = pkg_resources.resource_filename("gym_pybullet_drones", "assets")
    dest = os.path.join(assets_dir, "surveillance.urdf")

    if os.path.exists(dest):
        return  # already installed

    # Find our URDF relative to this file: ../../assets/surveillance.urdf
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(repo_root, "assets", "surveillance.urdf")

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"surveillance.urdf not found at {src}. "
            "Make sure assets/surveillance.urdf exists in the repo root."
        )

    shutil.copy2(src, dest)
    print(f"[patch] Installed surveillance.urdf → {dest}")


def _patch_enum():
    """Add SURVEILLANCE to DroneModel enum if not already present."""
    from gym_pybullet_drones.utils.enums import DroneModel

    if hasattr(DroneModel, "SURVEILLANCE"):
        return  # already patched

    # Dynamically add the enum member
    # This is equivalent to: SURVEILLANCE = "surveillance"
    new_member = "surveillance"
    DroneModel._value2member_map_[new_member] = DroneModel._value2member_map_.get(new_member)

    # Create a proper enum member
    obj = object.__new__(DroneModel)
    obj._name_ = "SURVEILLANCE"
    obj._value_ = new_member
    DroneModel._member_map_["SURVEILLANCE"] = obj
    DroneModel._value2member_map_[new_member] = obj
    DroneModel._member_names_.append("SURVEILLANCE")

    # Also write to the actual file so it persists across sessions
    enums_path = os.path.join(
        os.path.dirname(pkg_resources.resource_filename("gym_pybullet_drones", "utils")),
        "utils", "enums.py",
    )
    with open(enums_path, "r") as f:
        content = f.read()

    if "SURVEILLANCE" not in content:
        content = content.replace(
            '    RACE = "racer"  # Racer drone in the X configuration',
            '    RACE = "racer"  # Racer drone in the X configuration\n'
            '    SURVEILLANCE = "surveillance"  # 4kg surveillance quadrotor for tracking missions',
        )
        with open(enums_path, "w") as f:
            f.write(content)
        print("[patch] Added SURVEILLANCE to DroneModel enum")
