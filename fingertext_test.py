
from sklearn.cluster import KMeans
import string

import torch.backends.cudnn as cudnn

import craft_utils
import imgproc

from craft import CRAFT
from color_utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, ResizeNormalize
from model import Model
import math

from utils.datasets import *
from utils.smartmeter_modbus import *


import argparse


from PIL import Image

import torch

from torch.autograd import Variable


import cv2

import time


# All the possible words corresponding to the button/text-box number
words1 = ["Continue", "Load", "System...", "Head...", "Right", "Forward", "Up", "Set Network...", "Static IP...",
          "Increment", "Yes", "Start Model", "Pause",
          "Lights always on", "Lights normal", "Deutsch", "Resume"]
words2 = ["Material...", "Unload...", "Load Model", "Setup...", "Gantry...", "Left", "Backward", "Down", "Reverse",
          "Dynamic IP...", "Test Parts...", "Lights off",
          "Next Digit", "Disable UpnP", "Enable UpnP", "English", "Stop", "No"]
words3 = ["Standby Mode...", "Machine...", "Tip...", "Select Axis", "Select Drive", "Load Upgrade...", "Last Digit",
          "Select Language...", "Espanol", "Show Time"]
words4 = ["Maintenance...", "Done...", "Cancel", "Next...", "Auto Powerdown"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')
## #######################################################################################################
parser.add_argument('--image_folder', required=False, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='weights/TPS-ResNet-BiLSTM-Attn.pth',
                    help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

args = parser.parse_args()


net = CRAFT()  # initialize

# print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet

    refine_net = RefineNet()
    # print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

# Text Recognition
if args.sensitive:
    args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
args.num_gpu = torch.cuda.device_count()

""" model configuration """
if 'CTC' in args.Prediction:
    converter = CTCLabelConverter(args.character)
else:
    converter = AttnLabelConverter(args.character)
args.num_class = len(converter.character)

if args.rgb:
    args.input_channel = 3
model = Model(args)
# print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
#       args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
#       args.SequenceModeling, args.Prediction)
model = torch.nn.DataParallel(model).to(device)

# load model
# print('loading pretrained model from %s' % args.saved_model)
model.load_state_dict(torch.load(args.saved_model, map_location=device))

# predict
model.eval()


def demo(opt, roi, button=False):
    predict_list = []
    with torch.no_grad():
        batch_size = roi.size(0)
        image = roi.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        # log = open(f'./log_demo_result.txt', 'a')
        dashed_line = '-' * 80

        if button:
            head = f'{"predicted_labels":25s}\tconfidence score\tFinger On Button: TRUE'
        else:
            head = f'{"predicted_labels":25s}\tconfidence score\tFinger On Button: FALSE'

        # print(f'{dashed_line}\n{head}\n{dashed_line}')
        # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            # print(f'\t{pred:25s}\t{confidence_score:0.4f}')

            predict_list.append(pred)
        return (predict_list)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def worker_predict_image():
    counter = 1

    while True:

        print("Testing frame number: ", counter)

        frame = cv2.imread(f"finger_text_selected/camera{counter}.jpg", flags=cv2.IMREAD_COLOR)

        if frame is None:
            counter += 1
            continue


        start_time = time.time()

        _, this_frame = cv2.imencode('.jpg', frame)
        w, h, _ = frame.shape
        ee = np.zeros(frame.shape)

        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 200, 255, 0)
        edges = cv2.Canny(imgray, 150, 210, L2gradient=True)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        closed_contours = []

        for n, i in enumerate(contours):
            if cv2.contourArea(i) > cv2.arcLength(i, True) and cv2.contourArea(i) > (
                    w / 1080) ** 2 * 15000 and n % 2 == 0 and cv2.contourArea(i) < 10000:
                closed_contours.append(i)

        # for n, i in enumerate(contours):
        #     if cv2.contourArea(i) > cv2.arcLength(i, True) and cv2.contourArea(i) > 100:
        #         closed_contours.append(i)


        # cv2.drawContours(frame, closed_contours, -1, (0, 255, 0), 2)

        # Filter other bbox.
        width_list = np.asarray([np.max(i[:, :, 0]) - np.min(i[:, :, 0]) for i in closed_contours]).reshape(-1,
                                                                                                            1)
        final_predict = ""
        finger_predict = ""

        if len(width_list) != 0:
            if len(width_list) == 1:
                kmeans = KMeans(n_clusters=1, random_state=0).fit(width_list)
            else:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(width_list)

            kmeans_labels = kmeans.labels_.tolist()
            flag = None

            if kmeans_labels.count(0) > 1: flag = 0
            if kmeans_labels.count(1) > 1: flag = 1

            if kmeans_labels.count(0) > kmeans_labels.count(1):
                max_count = kmeans_labels.count(0)
            else:
                max_count = kmeans_labels.count(1)

            idx = np.where(kmeans.labels_ == flag)[0].tolist()
            filtered_contours = [con for i, con in enumerate(closed_contours) if i in idx]
            # print('**NEW FRAME********************************************************')

            finger_pos = -1

            # for con_id, con in enumerate(filtered_contours):
            for con_id, con in reversed(list(enumerate(reversed(filtered_contours)))):

                button_num = con_id + 1

                ############### Get the button coordinates using old method

                # Get the boundary limits of the contour corners
                xmin = np.min(con[:, :, 0])
                xmax = np.max(con[:, :, 0])
                ymin = np.min(con[:, :, 1])
                ymax = np.max(con[:, :, 1])


                # Old calculation of button coordinates
                button_xmin = int(xmin - 0.3 * (xmax - xmin))
                button_xmax = int(xmax - 1.16 * (xmax - xmin))
                button_ymin = int(ymin + 0.15 * (ymax - ymin))
                button_ymax = int(ymax - 0.11 * (ymax - ymin))

                button_width = button_xmax - button_xmin
                button_height = button_ymax-button_ymin

                if (button_height > 1.5 * button_width): # There is a distortion (i.e. an angle of the camera)


                    # The other coordinate for the min and max corners of the text regions
                    (xmin_, _) = np.where(con[:, :, 0] == xmin)
                    (xmax_, _) = np.where(con[:, :, 0] == xmax)
                    (ymin_, _) = np.where(con[:, :, 1] == ymin)
                    (ymax_, _) = np.where(con[:, :, 1] == ymax)

                    # The corner coordinates for the text region bounding boxes

                    top_corner_x = con[:, :, 0][ymin_[0]]
                    top_corner_y = ymin

                    left_corner_x = xmin
                    left_corner_y = con[:, :, 1][xmin_[0]]

                    right_corner_x = xmax
                    right_corner_y = con[:, :, 1][xmax_[0]]

                    bottom_corner_x = con[:, :, 0][ymax_[0]]
                    bottom_corner_y = ymax

                    # Draw contours for the text regions (when with angle)

                    pts = np.array([[top_corner_x, top_corner_y], [right_corner_x, right_corner_y], [bottom_corner_x, bottom_corner_y],  [left_corner_x, left_corner_y]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 255),2)

                    # The width and height of the text regions (approximated)

                    text_region_top_width = top_corner_x - left_corner_x
                    text_region_bottom_width = right_corner_x - bottom_corner_x

                    text_region_top_height = left_corner_y - top_corner_y
                    text_region_bottom_height = bottom_corner_y - right_corner_y

                    # Find the coordinates for the buttons (when with angle)

                    if text_region_top_height < text_region_top_width:  # Left-tilted View


                        new_button_top_x = left_corner_x - 0.15 * text_region_top_width
                        new_button_top_y = left_corner_y + 0.15 * text_region_top_height

                        new_button_left_x = left_corner_x - 0.3 * text_region_top_width
                        new_button_left_y = left_corner_y + 0.3 * text_region_top_height

                        new_button_right_x = bottom_corner_x - 0.15 * text_region_bottom_width
                        new_button_right_y = bottom_corner_y + 0.15 * text_region_bottom_height

                        new_button_bottom_x = bottom_corner_x - 0.3 * text_region_bottom_width
                        new_button_bottom_y = bottom_corner_y + 0.3 * text_region_bottom_height

                        # Draw contours for the buttons
                        pts = np.array([[new_button_top_x, new_button_top_y], [new_button_left_x, new_button_left_y], [new_button_bottom_x, new_button_bottom_y],
                                        [new_button_right_x, new_button_right_y]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                    else: # Right-tilted View

                        # The width and height of the text regions (approximated)

                        text_region_top_width = right_corner_x - top_corner_x
                        text_region_bottom_width = bottom_corner_x - left_corner_x

                        text_region_top_height = right_corner_y - top_corner_y
                        text_region_bottom_height = bottom_corner_y - left_corner_y

                        new_button_top_x = top_corner_x - 0.3 * text_region_top_width
                        new_button_top_y = top_corner_y - 0.3 * text_region_top_height

                        new_button_left_x = left_corner_x - 0.3 * text_region_top_width
                        new_button_left_y = left_corner_y - 0.3 * text_region_top_height

                        new_button_right_x = top_corner_x - 0.15 * text_region_bottom_width
                        new_button_right_y = top_corner_y - 0.15 * text_region_bottom_height

                        new_button_bottom_x = left_corner_x - 0.15 * text_region_bottom_width
                        new_button_bottom_y = left_corner_y - 0.15 * text_region_bottom_height

                        # Draw contours for the buttons
                        pts = np.array([[new_button_top_x, new_button_top_y], [new_button_left_x, new_button_left_y], [new_button_bottom_x, new_button_bottom_y],
                                        [new_button_right_x, new_button_right_y]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                else: # There is no angled distortion

                    # Draw contours for the text regions (normal without angle)
                    cv2.drawContours(frame, filtered_contours, -1, (0, 255, 255), 2)

                    # Draw contours for the buttons
                    pts = np.array([[button_xmin, button_ymin], [button_xmin, button_ymax], [button_xmax, button_ymax],
                                    [button_xmax, button_ymin]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                # The center pixel of the button box
                center_x = button_xmin + int(0.5 * (button_xmax - button_xmin))
                center_y = button_ymin + int(0.5 * (button_ymax - button_ymin))

                # The center pixel of the button box (upper to this button box)
                upper_center_x = center_x
                upper_center_y = center_y - int(2.1 * (button_ymax - button_ymin))

                button = False
                # print("ID is:", con_id)
                if button_num != 0:

                    # print("Analying button No.", button_num)

                    my_center_R = frame[center_y, center_x, 2]
                    up_center_R = frame[upper_center_y, upper_center_x, 2]

                    if my_center_R > up_center_R:
                        diff = my_center_R - up_center_R
                    else:
                        diff = up_center_R - my_center_R

                    if diff > 40:
                        button = True
                        finger_pos = button_num  # Finger is on this button
                    else:
                        button = False

                # Original implementation: use OpenCV rectangle to draw the bounding box
                # cv2.rectangle(frame, (button_xmin, button_ymin), (button_xmax, button_ymax), (0, 255, 0), 2)



                # region = np.array(frame[ymin + 5:ymax - 5, xmin + 5:xmax - 5])
                region = np.array(frame[ymin-5:ymax+5, xmin-5:xmax+5])

                try:
                    bboxes_text, polys_text, score_text = test_net(net, region,
                                                                   args.text_threshold, args.link_threshold,
                                                                   args.low_text, args.cuda, args.poly, refine_net)
                except:
                    continue

                if len(bboxes_text) != 0:
                    image_tensors = []
                    transform = ResizeNormalize((args.imgW, args.imgH))
                    for bbox in bboxes_text:
                        xxmin = int(np.min(bbox[:, 0]))
                        xxmax = int(np.max(bbox[:, 0]))
                        yymin = int(np.min(bbox[:, 1]))
                        yymax = int(np.max(bbox[:, 1]))

                        if xxmin < 0:
                            xxmin = 0
                        if xxmax < 0:
                            xxmax = 0
                        if yymin < 0:
                            yymin = 0
                        if yymax < 0:
                            yymax = 0

                        roi = np.array(region[yymin:yymax, xxmin:xxmax])
                        roi_pil = Image.fromarray(roi).convert('L')
                        image_tensors.append(transform(roi_pil))

                    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
                    predict_list = demo(opt=args, roi=image_tensors, button=button)

                    # Special Cases

                    predict_string = ' '.join(predict_list)

                    if predict_string == "model load":
                        predict_string = "load model"

                    # Find the corresponding string item based on the predicted string

                    if button_num == 1:
                        scores = []
                        for item in words1:
                            scores.append(similar(item, predict_string))

                        predict_string = words1[scores.index(max(scores))]

                    if button_num == 2:
                        scores = []
                        for item in words2:
                            scores.append(similar(item, predict_string))

                        predict_string = words2[scores.index(max(scores))]

                    if button_num == 3:
                        scores = []
                        for item in words3:
                            scores.append(similar(item, predict_string))

                        predict_string = words3[scores.index(max(scores))]

                    if button_num == 4:
                        scores = []
                        for item in words4:
                            scores.append(similar(item, predict_string))

                        predict_string = words4[scores.index(max(scores))]

                    final_predict = final_predict + ";" + predict_string

                    for bbox in bboxes_text:
                        bbox[:, 0] = bbox[:, 0] + xmin
                        bbox[:, 1] = bbox[:, 1] + ymin

                        poly = np.array(bbox).astype(np.int32).reshape((-1))
                        poly = poly.reshape(-1, 2)
                        cv2.polylines(frame, [poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2)

                else:

                    final_predict = final_predict + ";" + "EMPTY STRING"

        duration = time.time() - start_time

        # grid = cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 3)

        # cv2.drawContours(frame, filtered_contours, -1, (0, 255, 255), 2)
        cv2.imwrite(f"results/frame{counter}.jpg", frame) # Save file

        print("Duration: ",  duration)
        print(f"[Image No.{counter}]: ", final_predict)

        # cv2.imshow('image', frame)
        # cv2.waitKey(0)
        # cv2.imwrite("test.jpg", frame)
        counter += 1


if __name__ == "__main__":
    print("Finger-text detection testing...")
    worker_predict_image()
