

line = "spam\tWin a free ticket\tReply now\r\n"

def parse_labeled_line(line):
    clean_line = line.rstrip("\r\n")
    label, text = clean_line.split("\t", 1)
    return label, text

def main():
    label, text = parse_labeled_line(line)
    print(label)
    print(text)

if __name__ == '__main__':
    main()