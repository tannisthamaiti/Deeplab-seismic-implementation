import argparse
import os
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pandas as pd

import torchvision.utils as vutils
from core.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb
import numpy
import matplotlib.pyplot as plt

def patch_label_2d(model, img, patch_size, stride):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size/2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0)

    num_classes = 6
    output_p = torch.zeros([1, num_classes, h+2*ps, w+2*ps])

    # generate output:
    for hdx in range(0, h-patch_size+ps, stride):
        for wdx in range(0, w-patch_size+ps, stride):
            patch = img_p[hdx + ps: hdx + ps + patch_size,
                          wdx + ps: wdx + ps + patch_size]
            patch = patch.unsqueeze(dim=0)  # channel dim
            patch = patch.unsqueeze(dim=0)  # batch dim

            assert (patch.shape == (1, 1, patch_size, patch_size))

            model_output = model(patch)
            output_p[:, :, hdx + ps: hdx + ps + patch_size, wdx + ps: wdx +
                     ps + patch_size] += torch.squeeze(model_output.detach().cpu())

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir, model_name = os.path.split(args.model_path)
    # load model:
    model = torch.load(args.model_path)
    model = model.to(device)  # Send to GPU if available
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    splits = [args.split if 'both' not in args.split else 'test1', 'test2']
    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape

        if args.inline:
            i_list = list(range(irange))
            i_list = ['i_'+str(inline) for inline in i_list]
        else:
            i_list = []

        if args.crossline:
            x_list = list(range(xrange))
            x_list = ['x_'+str(crossline) for crossline in x_list]
        else:
            x_list = []

        list_test = i_list + x_list
        print(list_test)

        file_object = open(
            pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()

        test_set = SectionLoader(is_transform=True,
                                 split=split,
                                 augmentations=None)
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      num_workers=4,
                                      shuffle=False)

        running_metrics_split = runningScore(n_classes)
        pred_class = []
        pixel_acc = []
        class_acc = []
        mean_class_acc = []
        fre_weighted_iou = []
        mean_iou = []
        confusion_mat = []
        cls_iu = []


        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for i, (images, labels) in enumerate(test_loader):
                print(f'split: {split}, section: {i}')
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels

                outputs = patch_label_2d(model=model,
                                         img=images,
                                         patch_size=args.train_patch_size,
                                         stride=args.test_stride)

                pred = outputs.detach().max(1)[1].numpy()
                gt = labels.numpy()
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)
                pred_class.append(pred)
                pixel_acc.append(running_metrics_overall.get_scores()[0]["Pixel Acc: "])
                class_acc.append(running_metrics_overall.get_scores()[0]["Class Accuracy: "])
                mean_class_acc.append(running_metrics_overall.get_scores()[0]["Mean Class Acc: "])
                fre_weighted_iou.append(running_metrics_overall.get_scores()[0]["Freq Weighted IoU: "])
                mean_iou.append(running_metrics_overall.get_scores()[0]["Mean IoU: "])
                confusion_mat.append(running_metrics_overall.get_scores()[0]["confusion_matrix"])
                cls_iu.append(running_metrics_overall.get_scores()[1])

                #numbers = [0, 99, 149, 399, 499]
                numbers = range(100)
                if i in numbers:
                    tb_original_image = vutils.make_grid(
                        image_original[0][0], normalize=True, scale_each=True)
                    writer.add_image('original_image',
                                     tb_original_image, i)
                    #torchvision.transforms.ToPILImage()(tb_original_image) 
                    my_string = 'image_original_' + str(i)
                    original_image = tb_original_image.permute(1, 2, 0).numpy()
                    fig, ax = plt.subplots(figsize=(14, 8))
                    ax.imshow(original_image)
                    plt.savefig("test/original_image/{}.jpg".format(my_string)) #, img)
                    plt.close()
                   
                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original))
                    print(correct_label_decoded.shape)
                    fig, ax1 = plt.subplots(figsize=(14, 8))
                    ax1.imshow(correct_label_decoded)
                    my_string1 = 'correct_label_' + str(i)
                    plt.savefig("test/original_label/{}.jpg".format(my_string1)) #, img)
                    plt.close()

                    out = F.softmax(outputs, dim=1)
                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = vutils.make_grid(
                        confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(np.squeeze(prediction))
                    print(decoded.shape)
                    fig, ax2 = plt.subplots(figsize=(14, 8))
                    my_string2 = 'predicted_' + str(i)
                    #plt.imsave("test/predicted/{}.jpg".format(my_string2), decoded)
                    ax2.imshow(decoded)
                    #my_string2 = 'predicted_' + str(i)
                    plt.savefig("test/predicted/{}.png".format(my_string2)) #, img)
                    plt.close()

    pd.DataFrame(pixel_acc).to_csv("test/metrics/pixel_acc.csv", index=False)
    pd.DataFrame(class_acc).to_csv("test/metrics/class_acc.csv", index=False)
    pd.DataFrame(mean_class_acc).to_csv("test/metrics/mean_class_acc.csv",index=False)
    pd.DataFrame(fre_weighted_iou).to_csv("test/metrics/freq_weighted_iou.csv",index=False)
    pd.DataFrame(mean_iou).to_csv("test/metrics/mean_iou.csv", index=False)
    pd.DataFrame(cls_iu).to_csv("test/metrics/cls_iu.csv", index=False)

    for i in range(len(confusion_mat)):
        name = "test/metrics/confusion_matrix/confusion_matrix_" + str(i) + ".csv"
        pd.DataFrame(confusion_mat[i]).to_csv(name, index=False)

                    # uncomment if you want to visualize the different class heatmaps
                    # unary = outputs.cpu().detach()
                    # unary_max = torch.max(unary)
                    # unary_min = torch.min(unary)
                    # unary = unary.add((-1*unary_min))
                    # unary = unary/(unary_max - unary_min)

                    # for channel in range(0, len(class_names)):
                    #     decoded_channel = unary[0][channel]
                    #     tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
                    #     writer.add_image(f'test_classes/_{class_names[channel]}', tb_channel, i)

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()

        # Add split results to TB:
        writer.add_text(f'test__{split}/',
                        f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
        for cdx, class_name in enumerate(class_names):
            writer.add_text(
                f'test__{split}/', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)

        writer.add_text(
            f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
        writer.add_text(
            f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
        writer.add_text(f'test__{split}/',
                        f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)

        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    # Add split results to TB:
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0)
    for cdx, class_name in enumerate(class_names):
        writer.add_text(
            'test_final', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0)

    writer.add_text(
        'test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
    writer.add_text(
        'test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)

    print('--------------- FINAL RESULTS -----------------')
    print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names):
        print(
            f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')
    
    # Save confusion matrix: 
    confusion = score['confusion_matrix']
    np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ")

    writer.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='path/to/model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    parser.add_argument('--train_patch_size', nargs='?', type=int, default=99,
                        help='The size of the patches that were used for training.'
                        'This must be correct, or will cause errors.')
    parser.add_argument('--test_stride', nargs='?', type=int, default=10,
                        help='The size of the stride of the sliding window at test time. The smaller, the better the results, but the slower they are computed.')

    args = parser.parse_args()
    test(args)
