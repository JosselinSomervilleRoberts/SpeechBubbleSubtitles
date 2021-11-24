
#index of the start of the word with a character at position index in text
def start_of_current_word(text, index):
    if index == 0 or text[index-1] == ' ':
        return index
    else:
        return start_of_current_word(text, index-1)
#index of the end of the word with a character at position index in text
def end_of_current_word(text, index):
    if index == len(text) or text[index] == ' ':
        return index
    else:
        return end_of_current_word(text, index+1)

def drawText():
    """Draws the text inside the bubble"""
    #use cv2.getTextSize
    lines = "How dare you detective Diaz I am your superior officer! BOOOOOOOOOOONE "
    text_size = len(lines)
    width = 20
    nb_lines = text_size // width + 1
    print("Nb of lines: %s" % str(nb_lines))
    print(("Length of text: %s" % str(len(lines))))
    margin = 5

    text_per_line = [lines[(index_line * len(lines)//nb_lines):((index_line+1) * len(lines)//nb_lines)] for index_line in range(nb_lines)]
    text_size_last_line = len(text_per_line[-1])
    empty_space_at_the_end = (width - margin) - text_size_last_line
    
    print(text_per_line)
    for index_line in range(nb_lines):
        #print("Indexes: %s - %s" % (str(index_line * (len(lines)//nb_lines) ), str((index_line+1) * (len(lines)//nb_lines) )))
        #print("Indexes test: %s - %s" % (str((index_line * len(lines)//nb_lines) ), str(((index_line+1) * len(lines)//nb_lines) )))
        #print()
        pass
    for index_line in range(nb_lines):
        #put text of the line in the right spot, centering and stuff
        #if line starts with a space, remove it
        if text_per_line[index_line][0] == ' ':
            text_per_line[index_line] = text_per_line[index_line][1:]
        #if the line cuts a word in half, either put the word on the next line or add a '-'
        index_end_of_last_word = len(text_per_line[index_line])
        index_end_of_line = width - margin
        #print("Condtions: %s < %s and %s < %s" % (str(index_line), str(nb_lines-1), str(index_end_of_line), str(index_end_of_last_word)))
        if index_line < nb_lines - 1 and index_end_of_line < index_end_of_last_word:
            #print(" conditionzz")
            if empty_space_at_the_end > index_end_of_last_word - index_end_of_line:
                #print(" empty space")
                #put word on next line
                index_start_of_last_word = start_of_current_word(text_per_line[index_line], index_end_of_line)
                text_per_line[index_line+1] = text_per_line[index_line][index_start_of_last_word:] + text_per_line[index_line+1]
                text_per_line[index_line] = text_per_line[index_line][:index_start_of_last_word]
                empty_space_at_the_end = empty_space_at_the_end - index_end_of_last_word + index_end_of_line
            else:
                #add '-'
                text_per_line[index_line] += '-'
        print(text_per_line[index_line])

#drawText()

def cutLinesIntoWords(lines):
    """Takes lines (string) as input and returns the list of words"""
    list_of_words = [""]
    current_lines_index = 0
    while current_lines_index < len(lines):
        if lines[current_lines_index] == ' ':
            list_of_words.append("")
        else:
            list_of_words[-1] += lines[current_lines_index]
        current_lines_index += 1
    return list_of_words

print(cutLinesIntoWords("kzndkzd jzkdbjzkd bdezjbdedbejzkdb"))