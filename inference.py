#encoding=utf-8
import argparse
import os
from utils.datasets import *
from utils.torch_utils import select_device, time_synchronized
from utils.general import *
from utils.plots import plot_one_box
from models.experimental import attempt_load


def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple inference script for inference a yolov5 network.')
    parser.add_argument('--gpu', default='cpu', help='GPU to use', type=str)
    parser.add_argument('--model_path', default='/Users/hrpeng/Desktop/模型交付/明厨亮灶/MouseHeadChefhat13413_multiscale.pt', help='Model checkpoint file path', type=str)
    parser.add_argument('--image_dir', default='/Users/hrpeng/Desktop/test', help='Image files directory', type=str)
    parser.add_argument('--saved_dir', default='/Users/hrpeng/Desktop/test/det_res_0.01', help='result saved directroy', type=str)
    parser.add_argument('--image_size', default=640, help='Image long edge size', type=int)
    parser.add_argument('--score_threshold', default=0.01, help='score threshold to save', type=float)
    parser.add_argument('--max_det', help='maximum detection counts', default=100, type=int)
    return parser.parse_args()


def inference(args):
    gpu = args.gpu
    model_path = args.model_path
    image_dir = args.image_dir
    saved_dir = args.saved_dir
    image_size = args.image_size
    score_threshold = args.score_threshold
    max_det = args.max_det

    # Initialize
    device = select_device(gpu)
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)  # delete output folder
    os.makedirs(saved_dir)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = torch.load(model_path, map_location=device)['model'].float()  # load to FP32
    model = attempt_load(model_path, map_location=device)  # load FP32 model

    model.to(device).eval()
    if half:
        print("#" * 50)
        print("Using FP16.")
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(image_dir, img_size=image_size)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Set class2file
    class2file = dict()
    for cls_name in names:
        class2file[cls_name] = open(os.path.join(saved_dir, cls_name + ".txt"), "w")

    # Run inference
    t0 = time.time()
    all_time = 0
    img = torch.zeros((1, 3, image_size, image_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, _ in dataset:
        try:
            img_id = os.path.splitext(Path(path).name)[0]
            img_list = img.tolist()

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, score_threshold, 0.5)
            t3 = time_synchronized()
            k = 0

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                save_path = str(Path(saved_dir) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        xyxy_str = list(map(lambda x: str(x), xyxy))
                        class2file[names[int(cls)]].write(("$".join([img_id, str(float(conf)), *xyxy_str]) + '\n'))  # label format

                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                k+=1
                print(k)
                # Print time (inference + NMS)
                print('%sDone. pred: (%.3fs)' % (s, t2 - t1), 'nms: (%.3fs)' % (t3 - t2))
                all_time += (t3 - t1)

                # Save results (image with detections)
                save_path = os.path.splitext(save_path)[0] + ".jpg"
                cv2.imwrite(save_path, im0)
        except Exception as e:
            print(e.with_traceback())
            continue

    print('Done. (%.3fs)' % (time.time() - t0), 'average_time: (%.2fms)' % (all_time*1000/1766))


def main():
    args = parse_args()
    with torch.no_grad():
        inference(args)


if __name__ == '__main__':
    main()
