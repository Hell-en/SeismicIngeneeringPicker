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
    def get_source_num(file_name): # -> int: # SOU_X - source number
        # TYPE OF SGY IS  '../FB data/testArticleData/qqq_-0020_filtered.sgy'
        string = file_name
        string = string[:-4]  # get rid of '.sgy'
        print(string)
        num = 0
        string = re.sub(r'^.*?_', '', string) # delete before _
        string = string.replace(re.search(r'(?:_)(.*)', string).group(), '') ## delete after _
        print(string)

        if string[0] == '-':  # if starts with '-'
            num =  (-1)*int(string[1:].lstrip('0'))
        else:
            string = string.lstrip('0')
            if (string == ''): # был 0000 = 0
                num = 0
            else:
                num = int(string)
        return num
