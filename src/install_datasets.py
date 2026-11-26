from pathlib import Path

import kagglehub

# 课程个人第 2 题：美食分类。数据目录沿用竞赛名称。
output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw1-person"
path = kagglehub.competition_download('ml-hw1-person', output_dir=str(output_dir))
print("Path to competition files:", path)

# 课程个人第 1 题：葡萄酒回归。数据目录沿用历史名称。
output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw2-person"
path = kagglehub.competition_download('ml-hw2-personal', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw3-person"
path = kagglehub.competition_download('bagprice', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw4-person"
path = kagglehub.competition_download('character', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw1-group"
path = kagglehub.competition_download('ml-hw1-group', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw2-group"
path = kagglehub.competition_download('ml-hw2-group', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw3-group"
path = kagglehub.competition_download('face-buaa', output_dir=str(output_dir))
print("Path to competition files:", path)

output_dir = Path(__file__).resolve().parents[1] / "datasets" / "ml-hw4-group"
path = kagglehub.competition_download('document-buaa', output_dir=str(output_dir))
print("Path to competition files:", path)
