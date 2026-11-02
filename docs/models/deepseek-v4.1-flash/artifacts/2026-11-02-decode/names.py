import re
def nm(s):
    s = re.sub(r'^void ', '', s).replace('(anonymous namespace)::', '')
    depth = 0
    for i, ch in enumerate(s):
        if ch == '<': depth += 1
        elif ch == '>': depth -= 1
        elif ch == '(' and depth == 0: return s[:i]
    return s
