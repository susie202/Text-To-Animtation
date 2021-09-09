import os
import sys
sys.path.append('./face2video')

from options.test_options import TestOptions
import torch
from models import create_model
import data
import util.util as util
from tqdm import tqdm

def video_add_audio(name, audio_path, processed_file_savepath, video_path):
    cmd = ['ffmpeg', '-i', '\'' + os.path.join(processed_file_savepath, name + '.mp4') + '\'',
                     '-i', audio_path,
                     '-q:v 0',
                     '-strict -2',
                     video_path,
                     # '\'' + os.path.join(result_path, 'output' + '.mp4') + '\'',
                     '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

def img2video(dst_path, prefix, video_path):
    cmd = ['ffmpeg', '-i', '\'' + video_path + '/' + prefix + '%d.jpg'
           + '\'', '-q:v 0', '\'' + dst_path + '/' + prefix + '.mp4' + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

def del_folder(path):
    os.system('rm -r {}'.format(path))

def del_file(path):
    os.system('rm {}'.format(path))

def model_load(model_dir):
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.checkpoints_dir = model_dir
    opt.name = ""
    opt.meta_path_vox = "./misc/demo.csv"
    opt.results_dir = '../data/video_data/'
    opt.num_inputs=1
    torch.manual_seed(0)
    model = create_model(opt).cuda()
    model.eval()
    return model, opt

######### test 2

def inference_single_audio(opt, path_label, model ,result_path):
    #
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()

    idx = 0
    video_names = ['G_Pose_Driven_']
    save_paths = []
    for name in video_names:
        save_path = os.path.join(processed_file_savepath, name)
        util.mkdir(save_path)
        save_paths.append(save_path)
    for data_i in tqdm(dataloader):
        # print('==============', i, '===============')
        fake_image_original_pose_a, fake_image_driven_pose_a = model.forward(data_i, mode='inference')
        for num in range(len(fake_image_driven_pose_a)):
            util.save_torch_img(fake_image_driven_pose_a[num],
                        os.path.join(save_paths[0], video_names[0] + str(idx) + '.jpg'))
            idx += 1

    if opt.gen_video:
        for i, video_name in enumerate(video_names):
            img2video(processed_file_savepath, video_name, save_paths[i])
        video_add_audio('G_Pose_Driven_', dataloader.dataset.audio_path, processed_file_savepath,result_path)
    del_folder(os.path.join(processed_file_savepath, video_names[0]))
    del_file(os.path.join(processed_file_savepath, 'G_Pose_Driven_.mp4'))
    del_file(os.path.join(processed_file_savepath, 'ref_id_0.jpg'))
    print('results saved...' + processed_file_savepath)
    del dataloader
    return os.path.join(result_path, 'output' + '.mp4')


def inference(audio, image, model, opt, video_path):
    if 'sad' in image:
        pose_str = './data/pose_data/517600078 160'
    elif 'ang' in image:
        pose_str = './data/pose_data/517600078 160'
    else:
        pose_str = './data/pose_data/517600078 160'
    path_label = image+" 1 "+pose_str+" "+audio+" None 0 None"
    video_link = inference_single_audio(opt, path_label, model, video_path)
    return video_link