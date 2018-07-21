from video_decaptioning_data import VideoDecaptionData

def get_training_set(opt, spatial_transform, temporal_transform):

    if opt.dataset == 'VideoDecaptionData':
        training_data = VideoDecaptionData(
            opt.video_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            opt=opt)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform):

    if opt.dataset == 'VideoDecaptionData':
        validation_data = VideoDecaptionData(
            opt.video_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            sample_duration=125,
            opt=opt)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform):
    # assert opt.test_subset in ['val', 'test']

    # if opt.test_subset == 'val':
    #     subset = 'validation'
    # elif opt.test_subset == 'test':
    #     subset = 'testing'
    if opt.dataset == 'VideoDecaptionData':
        test_data = VideoDecaptionData(
            opt.video_path,
            'testing',
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            sample_duration=opt.sample_duration,
            opt=opt)

    return test_data