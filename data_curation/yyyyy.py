def image_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export/10087175377'
    data = 'CT.1.2.840.113619.2.55.3.380389780.37.1304454653.924.1.dcm'
    ds = dicom.read_file(proj_dir + '/' + data, force=True)
    print(ds)
    print(ds[0x0008, 0x103e])
    print(str(ds[0x0008, 0x103e]))
    if 'H+N SCAN/SCRAM' in str(ds[0x0008, 0x103e]):
        print('this is a head neck scan')
