import os
import re
import shutil

def get_file_name(path): #获取当前path文件或path目录的名称
    path,file_name = os.path.split(path)
    return file_name

def get_parent_path(path): #获取当前path文件或path目录的双亲目录的路径
    path,filename = os.path.split(path)
    return path

def remove_redundant_file(INPUT_FOLDER):    #删除文件夹内冗余的文件
    for dir_name , subdir_list , file_list in os.walk(INPUT_FOLDER):
        for file in file_list:
            if '.dcm' not in file.lower():
                file_path = os.path.join(dir_name,file)
                os.remove(file_path)
    return 

def remove_redundant_dir(INPUT_FOLDER): #递归删除所有空文件夹
    for dir_name , subdir_list , file_list in os.walk(INPUT_FOLDER):
        for subdir in subdir_list:
            if not os.listdir(os.path.join(dir_name,subdir)):
                os.removedirs(os.path.join(dir_name,subdir))
    return 

def move_dir1(INPUT_FOLDER): #把病历号移动到根目录下
    patientID_dir_list = []    #存储所有patientID文件夹的路径
    move_dir_list=[]    #存储patientID文件夹移动后的路径
    pattern1 = re.compile("^\d{6,8}$") #匹配名为6-8位数字的文件夹，后果后续patient文件夹名称pattern有更新，在此处更新
    pattern2 = re.compile("^\d{6,8}-\d{1,2}$") #匹配名如599993-2的文件夹
    for dir, subdir_list, file_list in os.walk(INPUT_FOLDER):
        dir_name = get_file_name(dir)   #对于当前dir_name，如果能够匹配patientID,说明该文件夹为patientID文件夹，加入patientID_dir_list
        if (re.match(pattern1,dir_name) != None or re.match(pattern2,dir_name) != None) and len(file_list) == 0:    #确保匹配到的不是dcm文件的上级目录
            patientID_dir_list.append(dir)
    for patient_dir_path in patientID_dir_list:
        patient_dir_name = get_file_name(patient_dir_path)
        move_path = os.path.join(INPUT_FOLDER, patient_dir_name)    #move_path为要移动到的路径
        move_dir_list.append(move_path)
        shutil.move(patient_dir_path, move_path)
    return move_dir_list

def move_dir2(INPUT_FOLDER):    #删除打开PatientID文件夹后例如20170913001265文件夹
    for dir, subdir_list, file_list in os.walk(INPUT_FOLDER):
        dir_name = get_file_name(dir)
        if len(file_list) != 0:
            move_path = os.path.join(INPUT_FOLDER,dir_name)
            if not os.path.exists(move_path): #处理当遇到一个病人有两个序列时，序列内部文件名冲突的情况
                shutil.move(dir, move_path)
            else:
                dir_name = dir_name + '_' + '2'
                move_path = os.path.join(INPUT_FOLDER,dir_name)
                if not os.path.exists(move_path):
                    shutil.move(dir, move_path)
                else:
                    dir_name = dir_name + '_' + '3'
                    move_path = os.path.join(INPUT_FOLDER,dir_name)
                    if not os.path.exists(move_path):
                        shutil.move(dir, move_path)
                    else:
                        print("patient :"  , dir_name , " has more than three series, plz del to one series!!!")
                        exit(0)
    return 

def movePosDisp(INPUT_FOLDER):  #删除文件名为PosDisp且文件夹内dcm文件数小于等于3的目录
    for dir_name , subdir_list , file_list in os.walk(INPUT_FOLDER):
        if 'PosDisp' in dir_name:
            if len(file_list) <= 3 : 
                shutil.rmtree(dir_name)
    return 