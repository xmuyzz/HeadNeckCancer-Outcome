def show_mri(img, mask):
    for i in range(img.shape[3]):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.axis(False)
        plt.imshow(img[0, :, :, i], cmap='gray')
        fig.add_subplot(1, 2, 2)
        plt.axis(False)
        plt.imshow(mask[0, :, :, i], cmap='gray')
        clear_output(wait=True)
        plt.show()
def show_mri_slice(img, mask, slice, pred = None):
    fig = plt.figure(figsize=(10, 7))
    if(pred is  None):
      fig.add_subplot(1, 2, 1)
      plt.axis(False)
      plt.imshow(img[0, :, :, slice], cmap='gray')
      fig.add_subplot(1, 2, 2)
      plt.axis(False)
      plt.imshow(mask[0, :, :, slice], cmap='gray')
    else:
      fig.add_subplot(1, 3, 1)
      plt.axis(False)
      plt.imshow(img[0, :, :, slice], cmap='gray')
      fig.add_subplot(1, 3, 2)
      plt.axis(False)
      plt.imshow(mask[0, :, :, slice], cmap='gray')
      fig.add_subplot(1, 3, 3)
      plt.axis(False)
      plt.imshow(pred[0, :, :, slice], cmap='gray')
    plt.show()

def check_data():
    img, mask = load_data(root_dir, "HEK_001.nii.gz", 'train', True)

    pred = np.concatenate([np.expand_dims(pred_1, axis = 0), np.expand_dims(pred_2, axis = 0)], axis = 0)

    print(img.shape, mask.shape, pred.shape)

    show_mri_slice(img, mask, 15, pred)

    show_mri(img, mask)
