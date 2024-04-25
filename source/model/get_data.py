import numpy as np
import pandas as pd
from dataio import write_nii, import_data_filename
from image_process import load_image_correct_oritation


def main():

    subject_list = import_data_filename('/home/zhao/Documents/cohort/JDC/', 'N4.nii.gz')

    output = 'rawdata/'
    for n, tmp in enumerate(subject_list):
        t2 = load_image_correct_oritation(subject_list[n])
        write_nii(t2, output+str(n)+'x.nii.gz')
        label = load_image_correct_oritation(subject_list[n][:-14]+'9tissue_labels.nii.gz')
        write_nii(label, output+str(n)+'y.nii.gz')
        mask = redefine_label(t2)
        write_nii(mask, output+str(n)+'m.nii.gz')


    sub = pd.read_excel('/home/zhao/Documents/cohort/r01.xlsx', sheet_name='fetal', skiprows=1)
    sub.columns =['ID','case','scan1','scan2','scan3']
    ID = []
    scan = []
    # mydata = pd.DataFrame(columns=['ID', 'GA'])
    file_handle = open(output+'ga.txt','w')
    for n, tmp in enumerate(subject_list):
        ID = int(tmp[39:43])
        scan = int(tmp[45:46])
        matchdata = sub.loc[sub['ID'] == ID]
        GA = matchdata.iloc[0, 1+scan]
        case = matchdata.iloc[0, 1]
        file_handle.write(str(int(GA))+'\n')
    file_handle.close()


def redefine_label(data):
    'manual redefine the labels'
    y = np.zeros(data.shape, np.float32)  # others labeld as 6
    y[data > 10] = 1  # 0 background
    return y


if __name__ == '__main__':
    main()
