import json
import shutil
from pathlib import Path

def clean_pose_data(pose_json_dir: Path,
                    bad_dir_name: str = "bad",
                    expected_kps: int = 33,
                    min_visibility_frac: float = 0.5):
    """
    Moves any pose-json into a "bad" subfolder if:
      - it doesn't have exactly expected_kps landmarks
      - fewer than min_visibility_frac of those landmarks have visibility > 0
    """
    pose_json_dir = Path(pose_json_dir)
    bad_dir = pose_json_dir / bad_dir_name
    bad_dir.mkdir(parents=True, exist_ok=True)

    for js in pose_json_dir.glob("*.json"):
        with open(js, "r") as f:
            data = json.load(f)
        kps = data.get("landmarks", [])
        # count how many keypoints are 'visible'
        visible_count = sum(1 for kp in kps if kp.get("visibility", 0) > 0)
        if len(kps) != expected_kps or visible_count < expected_kps * min_visibility_frac:
            target = bad_dir / js.name
            shutil.move(str(js), str(target))
            print(f"  ⚠️  Moved bad pose JSON → {target.relative_to(pose_json_dir.parent)}")
    
    print(f"all the images and there keypoint now is good :)")
    
    
# if __name__ == "__main__":
#     path = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Pose\Simples"
#     clean_pose_data(path)