'''
Reference
---------
https://github.com/icometrix/dicom2nifti
'''
import os
import os.path as osp
import dicom2nifti
import dicom2nifti.settings as settings
import SimpleITK as sitk
import nibabel as nib
import cv2


#toolpath="/home/lipeng/anaconda3/pkgs/dcm2niix-1.0.20210317-h4bd325d_0/bin"

def dcm_nifti(inputfile: str,outfile: str):
    '''
    convert dcm to .nii.gz
    inputfile
        - FU ZHENG XIANG(patient)
            - T1_Ax(modality)
                - xxxxxxxxxxx.dcm
                - xxxxxxxxxxx.dcm
                ...
            - T1_E_Ax(modality)
                - xxxxxxxxxxx.dcm
                - xxxxxxxxxxx.dcm
                ....
        - HU YI XUAN
        ...
    outfile
        - med
            -FU ZHENG XIANG(patient)
                - T1_Ax(modality)
                    -T1_Ax.nii.gz
                - T1_E_Ax(modality)
                    -T1_Ax.nii.gz
            - HU YI XUAN
            ...
    '''
    assert osp.exists(inputfile), 'dataset dir {} does not exist.'.format(
        inputfile)

    path='C:/Program Files/GDCM 3.0/bin/gdcmconv.exe'
    
    if not osp.exists(outfile):
        os.mkdir(outfile)
    
    outfile=outfile+'/med'
    if not osp.exists(outfile):
        os.mkdir(outfile)

    patient_list = os.listdir(inputfile)
    print(patient_list)
    for patient in patient_list:
        inputfile1 = inputfile + '/' + patient
        outfile1 = outfile + '/' + patient
        if not osp.exists(outfile1):
            os.mkdir(outfile1)
        modality_list = os.listdir(inputfile1)
        print(modality_list)

        for modality in modality_list:
            inputfile2 = inputfile1 + '/' + modality
            outfile2 = outfile1 + '/' + modality
            if not osp.exists(outfile2):
                os.mkdir(outfile2)
            settings.set_gdcmconv_path(path)
            dicom2nifti.convert_directory(inputfile2, outfile2, compression=True, reorient=True)
            #dcm2niix = toolpath + '/' + 'dcm2niix'

            #os.system('%s -o %s -z y %s' %(dcm2niix, outfile2, inputfile2))

        # 安装dcm2niix方法
       # https://github.com/ANTsX/ANTs/releases   


def  change_name(outputfile: str):
    assert osp.exists(outputfile), 'dataset dir {} does not exist.'.format(
        outputfile)
    patient_list = os.listdir(outputfile)
    print(patient_list)
    for patient in patient_list:
        outputfile1 = outputfile + '/' + patient
        modality_list = os.listdir(outputfile1)
        print(modality_list)
        for modality in modality_list:
            outputfile2 = outputfile1 + '/' + modality
            #print(outputfile2)
            data = (os.listdir(outputfile2))[0]
            os.rename(outputfile2+ '/' + data,outputfile2 + '/' + modality + '.nii.gz')

def  resize_data(outputfile: str,resize_file:str):
    assert osp.exists(outputfile), 'dataset dir {} does not exist.'.format(
        outputfile)
    if not osp.exists(resize_file):
        os.mkdir(resize_file)
    patient_list = os.listdir(outputfile)
    for patient in patient_list:
        outputfile1 = outputfile + '/' + patient
        resize_file1 = resize_file + '/' + patient
        if not osp.exists(resize_file1):
            os.mkdir(resize_file1)
        modality_list = os.listdir(outputfile1)
        for modality in modality_list:
            outputfile2 = outputfile1 + '/' + modality
            #print(outputfile2)
            resize_file2 = resize_file1 + '/' + modality
            if not osp.exists(resize_file2):
                os.mkdir(resize_file2)
            data = (os.listdir(outputfile2))[0]
            datafile = outputfile2 + '/' + data
            resize_datafile = resize_file2 + '/' + data
            img = sitk.ReadImage(datafile)
            img = sitk.GetArrayFromImage(img).astype(np.float64)
            img = img.transpose(2,1,0)
            # img = nib.load(datafile)
            # img = img.get_fdata()
            img = cv2.resize(img, (512, 512))
            print('patient:',patient,'modality:',modality,'size:',img.shape)
            # img_affine = img.affine()
            # nib.Nifti1Image(img, img_affine).to_filename(resize_datafile)
            img = img.transpose(2,1,0)
            img = sitk.GetImageFromArray(img)
            sitk.WriteImage(img,resize_datafile)

if __name__ == '__main__':
    # dcm_nifti('C:/Users/SYZX-Yjflash/Desktop/new_med_pre','C:/Users/SYZX-Yjflash/Desktop/new_med_process')
    # change_name('C:/Users/SYZX-Yjflash/Desktop/new_med_process/med')
    resize_data('C:/Users/SYZX-Yjflash/Desktop/new_med_process/med','C:/Users/SYZX-Yjflash/Desktop/new_med_process/med_resize')