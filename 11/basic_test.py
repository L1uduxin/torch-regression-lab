

records = [
    (301, "会议改到周四。"),
    (302, "领取限时礼品。"),
    (303, "明天交实验报告。"),
    (304, "领取限时礼品。"),
    (305, "会议改到周四。"),
]

def group_rows_by_text(records):
    groups = {}
    for row_id, text in records:
        if text not in groups:
            groups[text] = []
        groups[text].append(row_id)
    return groups


def find_duplicate_groups(records):
    group = group_rows_by_text(records)
    g = {}
    for text, row_ids in group.items():
        if(len(row_ids) > 1):
            g[text] = row_ids
    return g

def main():
    g = find_duplicate_groups(records)
    print(g)




if __name__ == '__main__':
    main()