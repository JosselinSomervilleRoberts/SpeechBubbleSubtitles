def jumpLinePosition(word):
    """Finds the position (if it exists) of a jump of line (= \\n)"""
    for i in range(len(word)-1):
        if "\\" in r"%r" % word[i:i+1] or "\\n" in r"%r" % word[i:i+1] or "\\N" in r"%r" % word[i:i+1]:
            return i
    return -1

print(jumpLinePosition(r"aladdio\Ndidi"))