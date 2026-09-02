import json
from validate_mission import *


def read_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data


def judge_type(mission: str) -> dict:
    submissions = mission.split(r'[，,]')
    print(submissions)
    submission_and_type = {}
    for submission in submissions:
        #Todo: 判断切分后的每句话属于哪种操作
        continue
    return submission_and_type


def process_ticket(src: str) -> bool:
    # 1.读取json字典
    data = read_json(src)

    # 2.切分并判断任务类型
    submission_and_type = judge_type(data["mission"])

    # 3.智能闭锁逻辑判断
    for item in submission_and_type:
        submission = item["submission"]
        type = item["type"]
        #Todo: 根据种类走对应的校验逻辑
    return True

src_route = "./command/220kV从庙变电站_核对35kV冉固线线路及3104开关在运行状态，现场具备操作条件_0901181553.json"
process_ticket(src_route)