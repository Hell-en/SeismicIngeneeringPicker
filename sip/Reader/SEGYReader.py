import glob
import re


class Reader:

    """Class for reading input files: .sgy and .txt with values of FB"""


    def __init__(self, path, pick):
        self.path = path
        self.pick = pick


    def get_file_names(self, txtfiles):
        for file in glob.glob("*.sgy"): # отсечь лишние файлы в папке
            txtfiles.append(file)


    @staticmethod
    def get_source_num(file_name):
        pattern = r'^[^_]*_'
        num = 0
        current = 0 # вспомогательная
        s = re.sub(pattern, '', s)
        if s.startswith('-'):
            num.append((-1)*int(s[1:5].lstrip('0')))
        else:
            current = s[0:4].lstrip('0')
            if current == '': num.append(0)
            else: num.append(int(s[0:4].lstrip('0')))
        return num