import os

def tokenize(start_index, end_index, input_text_content, conll_output_list, tag=None):
    if (start_index == end_index):
        return

    # split into tokens based on space
    tokens = input_text_content[start_index:end_index].split()

    if (tag is None):
        for token in tokens:
            if (token.isalpha() or token[0].isdigit()):
                conll_output_list.append(token + ' O' + '\n')
            else:
                cur_string = ''
                index = 0
                for ch in token:
                    if ((ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z')):
                        cur_string += ch
                    else:
                        # output the alphabetical token so far
                        if len(cur_string) > 0:
                            conll_output_list.append(cur_string + ' O' + '\n')
                            cur_string = ''
                        # special case to handle " ' "
                        if ch == "'":
                            conll_output_list.append(token[index:] + ' O' + '\n')
                            break
                        # every other special character is its own token
                        else:
                            conll_output_list.append(ch + ' O' + '\n')
                    index = index + 1

                if len(cur_string) > 0:
                    conll_output_list.append(cur_string + ' O' + '\n')
    else:
        conll_output_list.append(tokens[0] + ' B-' + tag + '\n')
        for t_index in range(1, len(tokens)):
            conll_output_list.append(tokens[t_index] + ' I-' + tag + '\n')

def get_filename(input_filepath):
    # assumes the file path has .txt extension
    return input_filepath[(input_filepath.rfind("/") + 1):-4]

def convert_to_conll(text_file_path, label_objects, output_file_path):
    input_file_name = get_filename(text_file_path)

    print("Processing for: " + input_file_name)

    # read the input file path
    input_file_path = os.path.join(os.getcwd(), text_file_path)

    conll_output_list = []
    offset_dict = {}

    # map all the offsetStarts to another dict
    for label in label_objects:
        cur_key = label['offsetStart']
        offset_dict[int(cur_key)] = [int(label['offsetEnd']), label['label']]

    print("Mapped {} out of {} labels".format(len(offset_dict.keys()), len(label_objects)))

    # read the text content
    with open(input_file_path, "r", newline="") as f:
        input_text_content = f.read()

    print("Successfully read the text file content")

    offset_dict_keys = list(offset_dict.keys())
    offset_dict_keys.sort()

    start_index = 0
    for offset in offset_dict_keys:
        tokenize(start_index, offset, input_text_content, conll_output_list)
        tokenize(offset, offset_dict[offset][0], input_text_content, conll_output_list, offset_dict[offset][1])
        start_index = offset_dict[offset][0]

    # tokenize remaining text
    tokenize(start_index, len(input_text_content), input_text_content, conll_output_list)

    print("Processing complete for: " + input_file_name)
    print('\n')

    # write conll to output file
    with open(output_file_path, "a") as f:
        f.writelines(conll_output_list)
        f.write('\n')
