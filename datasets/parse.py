import re

def split_file(src_input_file, tgt_input_file):
    with open(src_input_file, 'r', encoding='utf-8') as f_en:
        src_lines = f_en.readlines()
        
    with open(tgt_input_file, 'r', encoding='utf-8') as f_fr:
        tgt_lines = f_fr.readlines()

    src_sections = []
    tgt_sections = []
    src_current_section = []
    tgt_current_section = []

    # Regular expression to match lines that start with two consecutive capital letters
    # pattern = re.compile(r'^[A-Z]{3}.*')

    for i, src_line in enumerate(src_lines):
        # if pattern.match(src_line):
        if src_current_section and len(src_current_section) == 10:
            src_sections.append(src_current_section)
            tgt_sections.append(tgt_current_section)
            src_current_section = []
            tgt_current_section = []
        src_current_section.append(src_line)
        tgt_current_section.append(tgt_lines[i])

    # Add the last section if any
    if src_current_section and tgt_current_section:
        src_sections.append(src_current_section)
        tgt_sections.append(tgt_current_section)
        
    assert(len(src_sections) == len(tgt_sections))

    # Save each section to a separate file
    for j in range(len(src_sections)):
        with open(f"en-zh-short.en/{j}.txt", 'w', encoding='utf-8') as output_file:
            output_file.writelines(src_sections[j])
        with open(f"en-zh-short.zh/{j}.txt", 'w', encoding='utf-8') as output_file:
            output_file.writelines(tgt_sections[j])
        if j == 200:
            break        

if __name__ == "__main__":
    src_input_file = 'en-zh/News-Commentary.en-zh.en'
    tgt_input_file = 'en-zh/News-Commentary.en-zh.zh'
    split_file(src_input_file, tgt_input_file)