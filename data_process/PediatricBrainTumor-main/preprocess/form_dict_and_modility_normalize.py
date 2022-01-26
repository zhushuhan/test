#将dcm文件的series description规范化并生成series description2modility的字典
import xlrd

#加载series description 与 modility 对应的excel文件 并转化为{'series description' : 'modility'}的字典
def load_exce_and_form_dict(INPUT_EXCEL):  
    Ser_list = []
    Mod_list = []
    data = xlrd.open_workbook(INPUT_EXCEL)
    table = data.sheet_by_index(0)
    row = table.nrows
    col = table.ncols
    for i in range(1,row):
        Ser_list.append(table.cell(i,0).value)
        Mod_list.append(table.cell(i,1).value)
    dict={}
    len_list = len(Ser_list)
    for i in range(0,len_list):
        dict[Ser_list[i]] = Mod_list[i]
    return dict


#对series description进行切片操作，删除series description种多余的内容并对于有些modility含有: < > \ / * ? | "  导入无法以此种modility为名称命名文件夹的情况，特殊符号用空格进行代替
def modility_normalize(str,dict):    
    index = str.find("'") #对modility进行切片操作，删除series description种多余的内容
    str = str[index+1:-1]
    if 'CST' in str:    #如果当前字符串里有时间信息，例如 eADC:Dec 09 2020 11-57-58 CST 删除 :Dec 09 2020 11-57-58 CST‘  长度为25
        str=str[0:len(str)-25]
    if 'ADC (10^' in str: #对形如ADC (10^-6 mm/s):Apr 09 2019 11-32-17 CST 的进行特殊处理，否则会导致乱码无法打开生成的modility文件夹下的dcm文件
        str= 'ADC (10^-6 mms)'
    if ':' in str:
        str = str.replace(':','')
    if '<' in str:
        str = str.replace('<','')
    if '>' in str:
        str = str.replace('>','')
    if '\\' in str:
        str = str.replace('\\','')
    if '/' in str:
        str = str.replace('/','')
    if '*' in str:
        str = str.replace('*','')
    if '?' in str:
        str = str.replace('?','')
    if '|' in str:
        str = str.replace('|','')
    if '.' in str:
        str = str.replace('.','')
    # if 'PosDisp' in str:
    #     index1 = str.find(']')
    #     str = str[index1+2:]
    if str in dict.keys():
        str = dict[str]
    return str
