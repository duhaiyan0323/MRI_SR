import numpy as np
import SimpleITK as sitk

#-------------------------------------------------------------------------------
# resample sitk_image to referenced sitk_image
def resample_sitk_image_by_reference(reference, sitk_image):
    """
    :param reference:
    :param sitk_image:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetReferenceImage(reference)
    return resampler.Execute(sitk_image)

#------------------------------------------------------------------------------
# resampling the sitk_image with new size and spacing by preserving the spatial range
# appliable for 2d, 3d
def resample_sitk_image_center_aligned(sitk_image, size, range, padding='min'):
    """
    :param sitk_image:
    :param size:    tuple
    :param range:   tuple
    :param padding: 'zero', 'min'
    :return:
    """
    # checks
    if padding not in ['zero', 'min']: raise ValueError('padding should be either zero or min.')

    # get properties
    dim = sitk_image.GetDimension()
    np_old_size = np.array(sitk_image.GetSize(), dtype=np.uint32)
    np_old_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float)
    np_old_origin = np.array(sitk_image.GetOrigin(), dtype=np.float)

    # new spatial config
    if len(size) != len(range) != dim: raise ValueError('size or range is not matched with sitk_image.')
    np_range = np.array(range, dtype=np.float)
    np_size = np.array(size, dtype=np.float)
    np_spacing = np_range / np_size

    # faster performance : check whether to resample or not
    if np.equal(np_size, np_old_size).all() and np.equal(np_spacing, np_old_spacing).all(): return sitk_image

    new_size = tuple(np_size.astype(np.uint32).tolist())
    new_spacing = tuple(np_spacing.astype(np.float).tolist())

    # determine the new origin
    np_matrix_direction = np.array(sitk_image.GetDirection()).reshape(dim, dim)
    np_calibrated_shift = np.dot(np_matrix_direction, np_old_spacing * np_old_size / 2)
    np_center = np_old_origin + np_calibrated_shift
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_new_origin = np_center - np_calibrated_shift
    new_origin = tuple(np_new_origin.tolist())

    # determine the padding
    padding_value = 0.0
    if padding is 'min': padding_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitk_image)))

    # resample sitk_image into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitk_image, new_size, transform, sitk.sitkNearestNeighbor,
                         new_origin, new_spacing, sitk_image.GetDirection(),
                         padding_value, sitk_image.GetPixelID())

def resample_sitk_image_center_aligned2(sitk_image, size, range, padding='min'):
    """
    :param sitk_image:
    :param size:    tuple
    :param range:   tuple
    :param padding: 'zero', 'min'
    :return:
    """
    # checks
    if padding not in ['zero', 'min']: raise ValueError('padding should be either zero or min.')

    # get properties
    dim = sitk_image.GetDimension()
    np_old_size = np.array(sitk_image.GetSize(), dtype=np.uint32)
    np_old_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float)
    np_old_origin = np.array(sitk_image.GetOrigin(), dtype=np.float)

    # new spatial config
    if len(size) != len(range) != dim: raise ValueError('size or range is not matched with sitk_image.')
    np_range = np.array(range, dtype=np.float)
    np_size = np.array(size, dtype=np.float)
    np_spacing = np_range / np_size

    # faster performance : check whether to resample or not
    if np.equal(np_size, np_old_size).all() and np.equal(np_spacing, np_old_spacing).all(): return sitk_image

    new_size = tuple(np_size.astype(np.uint32).tolist())
    new_spacing = tuple(np_spacing.astype(np.float).tolist())

    # determine the new origin
    np_matrix_direction = np.array(sitk_image.GetDirection()).reshape(dim, dim)
    np_calibrated_shift = np.dot(np_matrix_direction, np_old_spacing * np_old_size / 2)
    np_center = np_old_origin + np_calibrated_shift
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_new_origin = np_center - np_calibrated_shift
    new_origin = tuple(np_new_origin.tolist())

    # determine the padding
    padding_value = 0.0
    if padding is 'min': padding_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitk_image)))

    # resample sitk_image into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitk_image, new_size, transform, sitk.sitkNearestNeighbor,
                         new_origin, new_spacing, sitk_image.GetDirection(),
                         padding_value, sitk_image.GetPixelID())

# ------------------------------------------------------------------------------
# resampling the sitk_image with new size and spacing by preserving the spatial range
# appliable for 2d, 3d
def resample_sitk_image_with_preserved_spatial_range(sitk_image,
                                                     size=None,
                                                     spacing=None,
                                                     padding='min'):
    """
    :param sitk_image:
    :param size:    tuple
    :param spacing: tuple
    :param padding: 'zero', 'min'
    :return:
    """
    # checks
    if sitk_image == None: raise ValueError('sitk_image should not be None.')
    checks = (size is not None, spacing is not None)
    if True not in checks: raise ValueError('either size or spacing should be specified.')
    if padding not in ['zero', 'min']: raise ValueError('padding should be either zero or min.')

    # get properties
    dim = sitk_image.GetDimension()
    np_old_size = np.array(sitk_image.GetSize(), dtype=np.uint32)
    np_old_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float)
    np_old_origin = np.array(sitk_image.GetOrigin(), dtype=np.float)

    # perserved range
    np_range = np.multiply(np_old_size, np_old_spacing)

    # calculate new size and new spacing
    if checks == (True, False):
        np_size = np.array(size, dtype=np.uint32)
        np_spacing = np.divide(np_range, np_size)
    elif checks == (False, True):
        np_spacing = np.array(spacing, dtype=np.float)
        np_size = np.divide(np_range, np_spacing)
    else:
        np_size = np_old_size
        np_spacing = np_old_spacing

    # faster performance : check whether to resample or not
    if np.equal(np_size, np_old_size).all() and np.equal(np_spacing, np_old_spacing).all(): return sitk_image

    new_size = tuple(np_size.astype(np.uint32).tolist())
    new_spacing = tuple(np_spacing.astype(np.float).tolist())

    # determine the new origin
    np_matrix_direction = np.array(sitk_image.GetDirection()).reshape(dim, dim)
    np_calibrated_shift = np.dot(np_matrix_direction, np_old_spacing * np_old_size / 2)
    np_center = np_old_origin + np_calibrated_shift
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_new_origin = np_center - np_calibrated_shift
    new_origin = tuple(np_new_origin.tolist())

    # determine the padding
    padding_value = 0.0
    if padding is 'min': padding_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitk_image)))

    # resample sitk_image into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitk_image, new_size, transform, sitk.sitkBSpline,
                         new_origin, new_spacing, sitk_image.GetDirection(),
                         padding_value, sitk_image.GetPixelID())

#------------------------------------------------------------------------------
# resampling the sitk_image with new size and spacing by preserving the spatial spacing
# applible for 2d, 3d
def resample_sitk_image_with_preserved_spatial_spacing(sitk_image,
                                                       size=None,
                                                       range=None,
                                                       padding='min'):
    """
    :param sitk_image:
    :param size:    tuple
    :param range:   tuple
    :param padding: 'zero', 'min'
    :return:
    """
    # checks
    if sitk_image == None: raise ValueError('sitk_image should not be None.')
    checks = (size is not None, range is not None)
    if True not in checks: raise ValueError('either size or spacing should be specified.')
    if padding not in ['zero', 'min']: raise ValueError('padding should be either zero or min.')

    # get properties
    dim = sitk_image.GetDimension()
    np_old_size = np.array(sitk_image.GetSize(), dtype=np.float)
    np_old_origin = np.array(sitk_image.GetOrigin(), dtype=np.float)

    # perserved spacing
    np_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float)

    # calculate new size and new range
    if checks == (True, False):
        np_size = np.array(size, dtype=np.float)
    elif checks == (False, True):
        np_range = np.array(range, dtype=np.float)
        np_size = np.divide(np_range, np_spacing)
    else:
        np_size = np_old_size

    # faster performance : check whether to resample or not
    if np.equal(np_size, np_old_size).all(): return sitk_image

    new_size = tuple(np_size.astype(np.uint32).tolist())
    new_spacing = tuple(np_spacing.astype(np.float32).tolist())

    # determine the new origin
    np_matrix_direction = np.array(sitk_image.GetDirection()).reshape(dim, dim)
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_old_size / 2)
    np_center = np_old_origin + np_calibrated_shift
    np_calibrated_shift = np.dot(np_matrix_direction, np_spacing * np_size / 2)
    np_new_origin = np_center - np_calibrated_shift
    new_origin = tuple(np_new_origin.tolist())

    # determine the padding
    padding_value = 0.0
    if padding is 'min': padding_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitk_image)))

    # resample sitk_image into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitk_image, new_size, transform, sitk.sitkNearestNeighbor,
                         new_origin, new_spacing, sitk_image.GetDirection(),
                         padding_value, sitk_image.GetPixelID())

#------------------------------------------------------------------------------
#
# test purpose
#
#------------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    root = '/home/hydu/MRI_SR/processMRI2/FLAIR/N4biascorrection/'
    save_root1 = '/home/hydu/MRI_SR/processMRI2/FLAIR/data/HR/'
    save_root2 = '/home/hydu/MRI_SR/processMRI2/FLAIR/data/LR/'
    dir_list=sorted(os.listdir(root))
    for i in range(len(dir_list)):
        
        nii_file = os.path.join(root,dir_list[i])
        sitk_img = sitk.ReadImage(nii_file)
        
        sitk_img_1mm = resample_sitk_image_with_preserved_spatial_range(sitk_img, spacing=(1, 1, 1))
        sitk_img_5mm = resample_sitk_image_with_preserved_spatial_range(sitk_img_1mm, spacing=(1, 1, 5))  #2mm or 5mm
        
        save_name_sr = save_root1 + dir_list[i].replace('N4.nii.gz', 'HR.nii.gz')
        save_name_lr = save_root2 + dir_list[i].replace('N4.nii.gz', 'LR.nii.gz')
        sitk.WriteImage(sitk_img_1mm, save_name_sr)
        sitk.WriteImage(sitk_img_5mm, save_name_lr)
        print(sitk.GetArrayFromImage(sitk_img_1mm).shape)
        print(sitk.GetArrayFromImage(sitk_img_5mm).shape)
        print(dir_list[i])
    print('DONE!')
    
    dir_list1=sorted(os.listdir(save_root1))
    dir_list2=sorted(os.listdir(save_root2))
    for i in range(len(dir_list2)):
        original_nii = os.path.join(save_root1,dir_list1[i])
        nii_file2 = os.path.join(save_root2,dir_list2[i])
        reference = sitk.ReadImage(original_nii)
        sitk_img2 = sitk.ReadImage(nii_file2)
        sitk_img_5mm_1mm = resample_sitk_image_by_reference(reference, sitk_img2)
        sitk.WriteImage(sitk_img_5mm_1mm, nii_file2.replace('.nii.gz', '.nii.gz'))
        print(sitk.GetArrayFromImage(reference).shape)
        print(sitk.GetArrayFromImage(sitk_img_5mm_1mm).shape)
