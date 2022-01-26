import pydicom
import os
import re
import shutil
import xlwt
import xlrd
from time import *
import form_dict_and_modility_normalize
import remove_redunant_file_and_dir


INPUT_FOLDER = "TIANTAN_HOSPITAL"  
INPUT_EXCEL = "Ser2Mod.xlsx"
EXCEL_SAVE_PATH="TIANTAN_HOSPITAL/modility_num.xls"  #excel表格保存路径

def get_file_name(path): #获取当前path文件或path目录的名称
    path,file_name = os.path.split(path)
    return file_name

def get_parent_path(path): #获取当前path文件或path目录的双亲目录的路径
    path,filename = os.path.split(path)
    return path

modility_list=[]    #全局变量，保存modility及其对应的病例数
modility_list_count = [0 for i in range (500)]
modility_num = []   #保存modility : patientNum
modility_type = 0    #全局变量，保存modility种类
dict = form_dict_and_modility_normalize.load_exce_and_form_dict(INPUT_EXCEL)

def rename_the_folder(INPUT_FOLDER):
    for dirname,subdirlist,filelist in os.walk(INPUT_FOLDER):   #遍历输入文件夹
        index=2
        print("current dirname: " , dirname) 
        for file in filelist:
            if '.dcm' in file.lower():  #如果当前文件为dcm
                file_path=os.path.join(dirname,file) #filepath为当前遍历到的dcm文件的绝对路径
                file_name=file[0:-4]    #保留当前dcm文件的名称，后续在遇到相同的series description需要移动时，会碰到同名文件
                ds=pydicom.dcmread(file_path,force=True)
                series_description = str(ds[0x0008,0x103E])
                modility=form_dict_and_modility_normalize.modility_normalize(series_description,dict)
                if modility not in modility_list:   #如果是一种新的modility，添加到modility列表，0用来统计每个modility出现的病例数
                    modility_list.append(modility)
                    global modility_type
                    modility_type+=1
                    print("this is a new modility:",modility)
                #else:
                    #print("modility ", modility , " is already exists")
                current_dirname = get_parent_path(file_path) #current_dirname为当前dcm文件所在目录
                grandparent_path =  get_parent_path(current_dirname)
                parent_path = os.path.join(grandparent_path,modility)   #parent_path为应该新建的文件夹的名称
                if current_dirname != parent_path:  #dirname为当前文件夹的名称，parent_path为要修改成的目标名称
                    if os.path.exists(parent_path): #如果目标文件夹已存在
                        file_rename = file_name + '_' + str(index) + '.dcm' #修改当前文件夹内dcm文件的名称，防止重复
                        file_move_path = os.path.join(parent_path,file_rename)  #file_move_path为更名后的dcm文件要移动到的位置
                        while os.path.exists(file_move_path):   #如果dcm文件要移动到的位置已存在, index+1
                            index+=1
                            file_rename = file_name + '_' + str(index) + '.dcm' #对dcm文件再进行一次重命名
                            file_move_path = os.path.join(parent_path,file_rename)
                        index=2
                        file_rename_path = os.path.join(current_dirname,file_rename)    #file_rename_path为更名后的dcm文件目前所在的真实位置
                        os.rename(file_path,file_rename_path) 
                        shutil.move(file_rename_path,parent_path)
                        if len(os.listdir(current_dirname)) == 0:
                            os.rmdir(current_dirname)
                            break
                    else:    
                        os.rename(dirname,parent_path)
                        break
                else:   #如果当前文件夹的名称和应该修改成的parent_path名称一致
                    break
        print('-----'*30)
    return  

def Calculate_num(INPUT_FOLDER):    #统计每一种Modility对应的病例个数
    for dirname, subdirlist,filelist in os.walk(INPUT_FOLDER):
        for i in range(0,modility_type):
            if modility_list[i] in subdirlist:
                modility_list_count[i] += 1
    return 

def write_excel_xls(path,value):    #存储excel表格到path
    index=len(value)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    for i in range(0,index):
        for j in range(len(value[i])):
            sheet.write(i,j,value[i][j])
    workbook.save(path)
    return 

begin_time = time()
remove_redunant_file_and_dir.remove_redundant_file(INPUT_FOLDER)
move_dir_list= remove_redunant_file_and_dir.move_dir1(INPUT_FOLDER)
for moved_dir in move_dir_list:
    remove_redunant_file_and_dir.move_dir2(moved_dir)
remove_redunant_file_and_dir.remove_redundant_dir(INPUT_FOLDER) 
rename_the_folder(INPUT_FOLDER)
Calculate_num(INPUT_FOLDER)
remove_redunant_file_and_dir.movePosDisp(INPUT_FOLDER) #因为所有包含PosDisp的目录内的dcm文件都小于等于三个，所以删除Posdisp目录，不列入统计列表

for i in range(0,modility_type):
    if 'PosDisp' not in modility_list[i]:  
        modility_num.append( [modility_list[i] , modility_list_count[i]] )

print("modility list: " , modility_list)
print("modility type num : " , modility_type)
write_excel_xls(EXCEL_SAVE_PATH,modility_num)
end_time=time()
run_time = end_time-begin_time
print("The process has finished, used time: " '%.2f' "s" %run_time)
