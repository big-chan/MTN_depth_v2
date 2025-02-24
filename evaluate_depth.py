from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from  torchvision.utils import save_image
from layers import disp_to_depth
from utils import readlines, sec_to_hm_str
from options import MonodepthOptions
import datasets
import networks
from tqdm import tqdm
print(torch.__version__)
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

import matplotlib.pyplot as plt
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
def viz_pre(feature):
    k = feature.shape[-1]  

    feature_image = feature.squeeze()#.transpose(0,1)
    return feature_image.numpy()
# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
def rank_error(errors, idx = 0, top = 5):
    list_err = []
    rank_list_maxi = []
    rank_list_mini = []
    errors = list(errors)
    for error in errors:
        list_err.append(list(error)[idx])
    copy_list_err = list_err.copy()
    list_err.sort(reverse=True)
    for value in list_err[:top]:
        rank_list_maxi.append(copy_list_err.index(value))
    print("maxi",rank_list_maxi)
    print(list_err[:top])
    for value in list_err[-top:]:
        rank_list_mini.append(copy_list_err.index(value))
    print("mini",rank_list_mini)
    print(list_err[-top:])
    return None
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        
        encoder_dict = torch.load(encoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
        decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        encoder = networks.test_hr_encoder.hrnet18(False)
        encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
        model_dict = encoder.state_dict()
        dec_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_model_dict})
        
        encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
        encoder.eval()
        depth_decoder.cuda() if torch.cuda.is_available() else depth_decoder.cpu()
        depth_decoder.eval()
        pred_disps = []
        print('-->Using\n cuda') if torch.cuda.is_available() else print('-->Using\n CPU')
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            init_time = time.time()
            i = 0 
            for data in dataloader:
                i += 1
                if torch.cuda.is_available():
                     
                    input_color = data[("color", 0, 0)].cuda()
                
                else:
                    input_color = data[("color", 0, 0)].cpu()
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp_0, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp_0.cpu()[:, 0].numpy()
                #pred_disp_viz = pred_disp_0.squeeze()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
            end_time = time.time()
            inferring = end_time - init_time
            print("===>total time:{}".format(sec_to_hm_str(inferring)))

        pred_disps = np.concatenate(pred_disps)
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
    #gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    
    tobe_cleaned = []
    cleaned = list(range(pred_disps.shape[0]))
    for i in tobe_cleaned:
        if i in cleaned:
            cleaned.remove(i)
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0
        
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    mean_errors = np.array(errors).mean(0)
    ## ranked_error
    ranked_error = rank_error(errors, 0 ,10)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    
def evaluate_with_train(encoder,depth_decoder,num_workers,data_path,eval_split,height,width,opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 50
    filenames = readlines(os.path.join(splits_dir, eval_split, "test_files.txt"))
    dataset = datasets.KAISTRAWDataset(data_path, filenames,height,width,[0], 4, is_train=False,thermal=opt.thermal)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)
    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []
    gt_depths = []
    print("-> Computing predictions with size {}x{}".format(
        width,height))

    with torch.no_grad():
        for data in tqdm(dataloader):
            
            if opt.thermal or opt.distill:
                input_color = data[("thermal", 0, 0)].cuda()
            else:    
                input_color = data[("color", 0, 0)].cuda()
#             input_color = torch.flip(input_color, [3])
            output = depth_decoder(encoder(input_color))
            
            pred_disp=output[("disp", 0)]
            if opt.scale_depth:
                pred_disp,_=disp_to_depth(pred_disp, opt.min_depth, opt.max_depth)
                
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            gt_depths.append(data["depth_gt"].squeeze().cpu().numpy())

            pred_disps.append(pred_disp)

    errors = []
    ratios = []

    for ii in tqdm(range(len(pred_disps))):
        for i in range(len(pred_disps[ii])):
            gt_depth = gt_depths[ii][i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[ii][i]
            pred_disp     = pred_disp#*10 #*1280
            if opt.softplus:
                pred_depth=np.log(np.exp(pred_disp)+1)
            else:
                pred_depth = 1 / pred_disp

            mask = gt_depth > 0
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    results_error={"abs_rel":mean_errors[0],"sq_rel":mean_errors[1],"rmse":mean_errors[2],\
                  "rmse_log":mean_errors[3],"a1":mean_errors[4],"a2":mean_errors[5],\
                  "a3":mean_errors[6]}
    return results_error

def evaluate_Kaist(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 50
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
    decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
    dataset = datasets.KAISTRAWDataset(opt.data_path, filenames,
                                       encoder_dict['height'], encoder_dict['width'],
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, scales=opt.scales,opt=opt)
    model_dict = encoder.state_dict()
    dec_model_dict = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_model_dict})

    encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
    encoder.eval()
    depth_decoder.cuda() if torch.cuda.is_available() else depth_decoder.cpu()
    depth_decoder.eval()
    pred_disps = []
    gt_depths = []
    print('-->Using\n cuda') if torch.cuda.is_available() else print('-->Using\n CPU')
    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))
        
    with torch.no_grad():
        init_time = time.time()
        i = 0 
        for data in tqdm(dataloader):
            i += 1
            if torch.cuda.is_available():

                input_color = data[("color", 0, 0)].cuda()

            else:
                input_color = data[("color", 0, 0)].cpu()
         
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
#             input_color=torch.flip(input_color, [3])
            output = depth_decoder(encoder(input_color))

            pred_disp_0, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp_0.cpu()[:, 0].numpy()
            
            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                
            gt_depth=data["depth_gt"].squeeze().cpu().numpy()
           
            disp=viz_pre(output[("disp", 0)][0].cpu()) 
            depth=viz_pre(data["depth_gt"][0].cpu()) 
            plt.imsave(os.path.join("disp.png"),disp,cmap="plasma")
            plt.imsave(os.path.join("gt.png"),depth,cmap="plasma")
            import pdb;pdb.set_trace()
            gt_depths.append(gt_depth)
            pred_disps.append(pred_disp)
            
        end_time = time.time()
        inferring = end_time - init_time
        print("===>total time:{}".format(sec_to_hm_str(inferring)))

    print("-> Evaluating")
    errors = []
    ratios = []
    
    tobe_cleaned = []
    for ii in tqdm(range(len(pred_disps))):
        for i in range(len(pred_disps[ii])):

            gt_depth = gt_depths[ii][i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[ii][i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
            mask = gt_depth > 0
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
    #         pred_depth *= opt.pred_depth_scale_factor

    #         if not opt.disable_median_scaling:
    #             ratio = np.median(gt_depth) / np.median(pred_depth)
    #             ratios.append(ratio)
    #             pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(gt_depth, pred_depth))

#     if not opt.disable_median_scaling:
#         ratios = np.array(ratios)
#         med = np.median(ratios)
#         print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    mean_errors = np.array(errors).mean(0)
    ## ranked_error
#     ranked_error = rank_error(errors, 0 ,10)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_Kaist(options.parse())
