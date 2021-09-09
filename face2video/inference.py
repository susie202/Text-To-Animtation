import os
import sys
sys.path.append('..')
from options.test_options import TestOptions
import torch
from models import create_model
import data
import util.util as util
from tqdm import tqdm



def video_add_audio(name, audio_path, processed_file_savepath):
    cmd = ['ffmpeg', '-i', '\'' + os.path.join(processed_file_savepath, name + '.mp4') + '\'',
                     '-i', audio_path,
                     '-q:v 0',
                     '-strict -2',
                     '\'' + os.path.join(processed_file_savepath, 'output' + '.mp4') + '\'',
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


def inference_single_audio(opt, path_label, model):
    #
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()
    # processed_file_savepath = '../data/video_data/'+kakao_id+"_"+text

    ## no use
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
        video_add_audio('G_Pose_Driven_', dataloader.dataset.audio_path, processed_file_savepath)
    del_folder(os.path.join(processed_file_savepath, video_names[0]))
    del_file(os.path.join(processed_file_savepath, 'G_Pose_Driven_.mp4'))
    del_file(os.path.join(processed_file_savepath, 'ref_id_0.jpg'))
    print('results saved...' + processed_file_savepath)
    del dataloader
    return


def main():

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.checkpoints_dir = "/home/whjung/Com2us/IntegratedCode/models/face2video"
    opt.name = ""
    opt.meta_path_vox = "./misc/demo.csv"
    opt.results_dir='../data/video_data/'
    torch.manual_seed(0)
    model = create_model(opt).cuda()
    model.eval()

    with open(opt.meta_path_vox, 'r') as f:
        lines = f.read().splitlines()

    for clip_idx, path_label in enumerate(lines):
        try:
            assert len(path_label.split()) == 8, path_label
            path_label = '../data/card_data/10172/ang 1 ../data/pose_data/517600078 160 ../data/voice_data/0000_안녕하세요.wav ../data/mouth_data/681600002 363 dummy'
            inference_single_audio(opt, path_label, model)

        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(path_label + '\n')
            print(str(ex))


def predict():
    opt = TestOptions().parse()
    opt.isTrain = False
    torch.manual_seed(0)
    model = create_model(opt).cuda()
    model.eval()

    with open(opt.meta_path_vox, 'r') as f:
        lines = f.read().splitlines()

    for clip_idx, path_label in enumerate(lines):
        try:
            assert len(path_label.split()) == 8, path_label
            inference_single_audio(opt, path_label, model)

        except Exception as ex:
            import traceback
            traceback.print_exc()



if __name__ == '__main__':
    main()
