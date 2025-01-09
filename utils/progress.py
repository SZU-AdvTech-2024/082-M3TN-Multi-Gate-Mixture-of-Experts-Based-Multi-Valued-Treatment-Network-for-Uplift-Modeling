import time

# 一种以结构化方式格式化文本输出的实用程序，使其更容易阅读
class WorkSplitter(object):
    def __init__(self):
        self.columns = 50
    # 打印节标题
    def section(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '='*self.columns+'\n' \
                 + "|"+' '*(left_length-1)+name+' '*(right_length-1)+'|\n'\
                 + '='*self.columns+'\n'

        print(output)
    # 打印小节标题
    def subsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '#' * (left_length-1) + ' ' + name + ' ' + '#' * (right_length-1) + '\n'
        print(output)
    # 打印子部分标题
    def subsubsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = self.columns-name_length-left_length

        output = '-' * (left_length-1) + ' ' + name + ' ' + '-' * (right_length-1) + '\n'
        print(output)


def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))