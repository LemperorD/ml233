from pathlib import Path

import kagglehub

kagglehub.login()

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw1-person"
path = kagglehub.competition_download('ml-hw1-person', output_dir=str(output_dir))

print("Path to competition files:", path)
