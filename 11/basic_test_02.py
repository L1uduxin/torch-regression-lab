
from basic_text_01 import parse_labeled_line
from basic_test import find_duplicate_groups,group_rows_by_text

lines = [
    "spam\tFree entry\tToday only\r\n",
    "ham\tCall me at six.\n",
    "spam\tFree entry\tToday only\n",
]
label_to_id = {"ham": 0, "spam": 1}

def build_records(lines):
    # 返回列表，每条记录为 (row_id, text, y)
    list = []
    for row_id, line in enumerate(lines, start=1):
        label, text = parse_labeled_line(line)

        list.append((row_id, text, label_to_id[label]))
    return list


def to_text_rows(records):
    # 输入：三元组列表 (row_id, text, y)
    # 返回：二元组列表 (row_id, text)
    group = []
    for item in records:
        group.append((item[0], item[1]))

    return group

def has_label_conflict(labels):
    # 返回布尔值 True 或 False，表示是否存在标签冲突
    if len(set(labels)) == 1 :
        return False
    else:
        return True


records = [
    (61, "下午开会。", 0),
    (62, "点击确认资格。", 1),
    (63, "请带实验记录。", 0),
    (64, "下午开会。", 0),
    (65, "点击确认资格。", 0),
    (66, "点击确认资格。", 1),
]

def find_label_conflicts(records):
    # 返回存在标签冲突的正文及其全部原编号
    result = {}

    text_list = set([data[1] for data in records])
    for text in text_list:
        label = [data[2] for data in records if data[1] == text]
        has_conflict = has_label_conflict(label)
        if has_conflict:
            ids = [record[0] for record in records if record[1] == text]
            result[text] = ids
    return result




def main():
    records = build_records(lines)
    group = to_text_rows(records)
    result = find_duplicate_groups(group)
    print(result)

if __name__ == '__main__':
    #main()
    #print(has_label_conflict([1,1,1]))
    #print(has_label_conflict([0,1,0]))
    result = find_label_conflicts(records)
    print(result)