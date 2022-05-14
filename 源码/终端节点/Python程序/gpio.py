#!/usr/bin/python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from tkinter import *

blue_key = 31
yellow_key = 35
black_key = 37
led_r = 11
led_g = 15
led_b = 7
fmq = 13

GPIO.setmode(GPIO.BOARD)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(37, GPIO.IN)
GPIO.output(13, GPIO.HIGH)
GPIO.setup(led_r , GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(31, GPIO.IN)
GPIO.setup(35, GPIO.IN)


while 1:
    while GPIO.input(blue_key) == GPIO.HIGH:
        GPIO.output(11, False)
        GPIO.output(15, True)
        time.sleep(1)
        GPIO.output(15, False)
        GPIO.output(7, True)
        time.sleep(1)
        GPIO.output(7, False)
        GPIO.output(fmq, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(11, True)
        time.sleep(1)
        def nothing(value):
            pass
        hsv_l = np.array([104, 81, 60])
        hsv_u = np.array([255, 255, 255])

        RTSP_URL = 'http://169.254.177.18:8080/?action=stream'  # your camera's rtsp url

        # 初期化USB摄像头
        cap = cv2.VideoCapture(RTSP_URL)
        if (cap.isOpened()):
            print("cap.isOpened() ")
            # setup_trackbars()
        Y_min = []
        for i in range(0, 40):
            ret, frame = cap.read()
            if not ret:
                break
                print("2")
            frame = cv2.resize(frame, (480, 360))
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # hl,sl,vl,hu,su,vu =get_trackbar_values()
            mask = cv2.inRange(img_hsv, hsv_l, hsv_u)
            mask_morph = mask.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # 开运算
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
            # 闭运算
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)
            output = cv2.bitwise_and(frame, frame, mask=mask_morph)

            cnts = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), raduis) = cv2.minEnclosingCircle(c)
                # 计算轮廓的矩
                M = cv2.moments(c)
                # 计算轮廓的重心
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # 计算坐标
                cx = int(M['m10'] / M['m00'])  # 求x坐标
                cy = int(M['m01'] / M['m00'])  # 求y坐标
                # 矩形
                w, h = 25, 25
                if raduis > 5:
                    # # 画出最小外接圆
                    # cv2.circle(frame, (int(x), int(y)), int(raduis), (0, 255, 255), 2)
                    # 矩形
                    cv2.rectangle(frame, (int(x) - int(raduis), int(y) - int(raduis), int(2 * raduis), int(2 * raduis)),
                                  color=(0, 0, 255), thickness=1)  # BGR
                    # 画出重心
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # 如果满足条件，就画出圆，画图函数，frame，中心，半径，颜色，厚度
                # if raduis>10:
                #     cv2.circle(frame,(int(x),int(y)),int(raduis),(0,255,0),6)
                cv2.imshow("Measurement", frame)
                cv2.moveWindow("Measurement", 1200, 50)
                Y_min.append(cy)
                if cv2.waitKey(300) & 0xFF is ord("q"):
                    break
        Y_min.sort(reverse=True)
        L = (Y_min[0] - 479) * 21 / 92 + 142.3087 - 3
        print(L, "cm")
        GPIO.output(7, True)
        time.sleep(1)
        GPIO.output(7, False)
        GPIO.output(13, GPIO.LOW)
        time.sleep(3)
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(11, True)
        time.sleep(1)
        GPIO.output(11, False)
        GPIO.output(15, True)
        time.sleep(1)
        GPIO.output(15, False)
        cap.release()
        cv2.destroyAllWindows()

        top = Tk()
        top.geometry("400x200")
        top.title("Measurement in progress")
        li = ['#Slowly loud == ending#', '#fastloud == start#', 'cm', L, 'Length = ,']
        listb = Listbox(top)
        for item in li:
            listb.insert(0, item)
        listb.pack()
        top.mainloop()
     while GPIO.input(35) == GPIO.HIGH:
         def detect(save_img=False):
             source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
             webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                 ('rtsp://', 'rtmp://', 'http://'))

             # Directories
             save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
             (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

             # Initialize
             set_logging()
             device = select_device(opt.device)
             half = device.type != 'cpu'  # half precision only supported on CUDA

             # Load model
             model = attempt_load(weights, map_location=device)  # load FP32 model
             imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
             if half:
                 model.half()  # to FP16

             # Second-stage classifier
             classify = False
             if classify:
                 modelc = load_classifier(name='resnet101', n=2)  # initialize
                 modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                     device).eval()

             # Set Dataloader
             vid_path, vid_writer = None, None
             if webcam:
                 view_img = True
                 cudnn.benchmark = True  # set True to speed up constant image size inference
                 dataset = LoadStreams(source, img_size=imgsz)
             else:
                 save_img = True
                 dataset = LoadImages(source, img_size=imgsz)

             # Get names and colors
             names = model.module.names if hasattr(model, 'module') else model.names
             colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

             # Run inference
             t0 = time.time()
             img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
             _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
             for path, img, im0s, vid_cap in dataset:
                 img = torch.from_numpy(img).to(device)
                 img = img.half() if half else img.float()  # uint8 to fp16/32
                 img /= 255.0  # 0 - 255 to 0.0 - 1.0
                 if img.ndimension() == 3:
                     img = img.unsqueeze(0)

                 # Inference
                 t1 = time_synchronized()
                 pred = model(img, augment=opt.augment)[0]

                 # Apply NMS
                 pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                            agnostic=opt.agnostic_nms)
                 t2 = time_synchronized()

                 # Apply Classifier
                 if classify:
                     pred = apply_classifier(pred, modelc, img, im0s)

                 # Process detections
                 for i, det in enumerate(pred):  # detections per image
                     if webcam:  # batch_size >= 1
                         p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                     else:
                         p, s, im0 = Path(path), '', im0s

                     save_path = str(save_dir / p.name)
                     txt_path = str(save_dir / 'labels' / p.stem) + (
                         '_%g' % dataset.frame if dataset.mode == 'video' else '')
                     s += '%gx%g ' % img.shape[2:]  # print string
                     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                     if len(det):
                         # Rescale boxes from img_size to im0 size
                         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                         # Print results
                         for c in det[:, -1].unique():
                             n = (det[:, -1] == c).sum()  # detections per class
                             s += '%g %ss, ' % (n, names[int(c)])  # add to string

                         # Write results
                         for *xyxy, conf, cls in reversed(det):
                             if save_txt:  # Write to file
                                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                     -1).tolist()  # normalized xywh
                                 line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                 with open(txt_path + '.txt', 'a') as f:
                                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                             if save_img or view_img:  # Add bbox to image
                                 label = '%s %.2f' % (names[int(cls)], conf)
                                 plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                     # Print time (inference + NMS)
                     print('%sDone. (%.3fs)' % (s, t2 - t1))

                     # Stream results
                     if view_img:
                         cv2.imshow(str(p), im0)
                         if cv2.waitKey(1) == ord('q'):  # q to quit
                             raise StopIteration

                     # Save results (image with detections)
                     if save_img:
                         if dataset.mode == 'images':
                             cv2.imwrite(save_path, im0)
                         else:
                             if vid_path != save_path:  # new video
                                 vid_path = save_path
                                 if isinstance(vid_writer, cv2.VideoWriter):
                                     vid_writer.release()  # release previous video writer

                                 fourcc = 'mp4v'  # output video codec
                                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                             vid_writer.write(im0)

             if save_txt or save_img:
                 s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                 print(f"Results saved to {save_dir}{s}")

             print('Done. (%.3fs)' % (time.time() - t0))
         if __name__ == '__main__':
             parser = argparse.ArgumentParser()
             parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
             parser.add_argument('--source', type=str, default='data/images',
                                 help='source')  # file/folder, 0 for webcam
             parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
             parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
             parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
             parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
             parser.add_argument('--view-img', action='store_true', help='display results')
             parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
             parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
             parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
             parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
             parser.add_argument('--augment', action='store_true', help='augmented inference')
             parser.add_argument('--update', action='store_true', help='update all models')
             parser.add_argument('--project', default='runs/detect', help='save results to project/name')
             parser.add_argument('--name', default='exp', help='save results to project/name')
             parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
             opt = parser.parse_args()
             print(opt)
             with torch.no_grad():
                 if opt.update:  # update all models (to fix SourceChangeWarning)
                     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                         detect()
                         strip_optimizer(opt.weights)
                 else:
                     detect()

         time.sleep(1)
      while GPIO.input(31) == GPIO.HIGH:
          def train(hyp, opt, device, tb_writer=None, wandb=None):
              logger.info(f'Hyperparameters {hyp}')
              save_dir, epochs, batch_size, total_batch_size, weights, rank = \
                  Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

              # Directories
              wdir = save_dir / 'weights'
              wdir.mkdir(parents=True, exist_ok=True)  # make dir
              last = wdir / 'last.pt'
              best = wdir / 'best.pt'
              results_file = save_dir / 'results.txt'

              # Save run settings
              with open(save_dir / 'hyp.yaml', 'w') as f:
                  yaml.dump(hyp, f, sort_keys=False)
              with open(save_dir / 'opt.yaml', 'w') as f:
                  yaml.dump(vars(opt), f, sort_keys=False)

              # Configure
              plots = not opt.evolve  # create plots
              cuda = device.type != 'cpu'
              init_seeds(2 + rank)
              with open(opt.data) as f:
                  data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
              with torch_distributed_zero_first(rank):
                  check_dataset(data_dict)  # check
              train_path = data_dict['train']
              test_path = data_dict['val']
              nc, names = (1, ['item']) if opt.single_cls else (
              int(data_dict['nc']), data_dict['names'])  # number classes, names
              assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

              # Model
              pretrained = weights.endswith('.pt')
              if pretrained:
                  with torch_distributed_zero_first(rank):
                      attempt_download(weights)  # download if not found locally
                  ckpt = torch.load(weights, map_location=device)  # load checkpoint
                  if hyp.get('anchors'):
                      ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
                  model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
                  exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
                  state_dict = ckpt['model'].float().state_dict()  # to FP32
                  state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
                  model.load_state_dict(state_dict, strict=False)  # load
                  logger.info(
                      'Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
              else:
                  model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

              # Freeze
              freeze = []  # parameter names to freeze (full or partial)
              for k, v in model.named_parameters():
                  v.requires_grad = True  # train all layers
                  if any(x in k for x in freeze):
                      print('freezing %s' % k)
                      v.requires_grad = False

              # Optimizer
              nbs = 64  # nominal batch size
              accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
              hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

              pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
              for k, v in model.named_modules():
                  if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                      pg2.append(v.bias)  # biases
                  if isinstance(v, nn.BatchNorm2d):
                      pg0.append(v.weight)  # no decay
                  elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                      pg1.append(v.weight)  # apply decay

              if opt.adam:
                  optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
              else:
                  optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

              optimizer.add_param_group(
                  {'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
              optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
              logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
              del pg0, pg1, pg2

              # Scheduler https://arxiv.org/pdf/1812.01187.pdf
              # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
              lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
              scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
              # plot_lr_scheduler(optimizer, scheduler, epochs)

              # Logging
              if wandb and wandb.run is None:
                  opt.hyp = hyp  # add hyperparameters
                  wandb_run = wandb.init(config=opt, resume="allow",
                                         project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                         name=save_dir.stem,
                                         id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)

              # Resume
              start_epoch, best_fitness = 0, 0.0
              if pretrained:
                  # Optimizer
                  if ckpt['optimizer'] is not None:
                      optimizer.load_state_dict(ckpt['optimizer'])
                      best_fitness = ckpt['best_fitness']

                  # Results
                  if ckpt.get('training_results') is not None:
                      with open(results_file, 'w') as file:
                          file.write(ckpt['training_results'])  # write results.txt

                  # Epochs
                  start_epoch = ckpt['epoch'] + 1
                  if opt.resume:
                      assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                      weights, epochs)
                  if epochs < start_epoch:
                      logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                                  (weights, ckpt['epoch'], epochs))
                      epochs += ckpt['epoch']  # finetune additional epochs

                  del ckpt, state_dict

              # Image sizes
              gs = int(max(model.stride))  # grid size (max stride)
              imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

              # DP mode
              if cuda and rank == -1 and torch.cuda.device_count() > 1:
                  model = torch.nn.DataParallel(model)

              # SyncBatchNorm
              if opt.sync_bn and cuda and rank != -1:
                  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
                  logger.info('Using SyncBatchNorm()')

              # EMA
              ema = ModelEMA(model) if rank in [-1, 0] else None

              # DDP mode
              if cuda and rank != -1:
                  model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

              # Trainloader
              dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                      hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                                      rank=rank,
                                                      world_size=opt.world_size, workers=opt.workers,
                                                      image_weights=opt.image_weights)
              mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
              nb = len(dataloader)  # number of batches
              assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
              mlc, nc, opt.data, nc - 1)

              # Process 0
              if rank in [-1, 0]:
                  ema.updates = start_epoch * nb // accumulate  # set EMA updates
                  testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                                 hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                                 rank=-1, world_size=opt.world_size, workers=opt.workers)[
                      0]  # testloader

                      # Anchors
                      if not opt.noautoanchor:
                          check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

              # Model parameters
              hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
              model.nc = nc  # attach number of classes to model
              model.hyp = hyp  # attach hyperparameters to model
              model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
              model.load_state_dict(s)
              model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
              model.names = names

              # Start training
              t0 = time.time()
              results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
              scheduler.last_epoch = start_epoch - 1  # do not move
              scaler = amp.GradScaler(enabled=cuda)
              logger.info('Image sizes %g train, %g test\n'
                          'Using %g dataloader workers\nLogging results to %s\n'
                          'Starting training for %g epochs...' % (
                          imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))
              for epoch in range(start_epoch,
                                 epochs):  # epoch ------------------------------------------------------------------
                  model.train()

                  # Update image weights (optional)
                  if opt.image_weights:
                      # Generate indices
                      if rank in [-1, 0]:
                          cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                          iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                          dataset.indices = random.choices(range(dataset.n), weights=iw,
                                                           k=dataset.n)  # rand weighted idx
                      # Broadcast if DDP
                      if rank != -1:
                          indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                          dist.broadcast(indices, 0)
                          if rank != 0:
                              dataset.indices = indices.cpu().numpy()

                  # Update mosaic border
                  # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
                  # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

                  mloss = torch.zeros(4, device=device)  # mean losses
                  if rank != -1:
                      dataloader.sampler.set_epoch(epoch)
                  pbar = enumerate(dataloader)
                  logger.info(
                      ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
                  if rank in [-1, 0]:
                      pbar = tqdm(pbar, total=nb)  # progress bar
                  optimizer.zero_grad()
                  for i, (imgs, targets, paths,
                          _) in pbar:  # batch -------------------------------------------------------------
                      ni = i + nb * epoch  # number integrated batches (since train start)
                      imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                      # Warmup
                      if ni <= nw:
                          xi = [0, nw]  # x interp
                          # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                          accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                          for j, x in enumerate(optimizer.param_groups):
                              # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                              x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0,
                                                           x['initial_lr'] * lf(epoch)])
                              if 'momentum' in x:
                                  x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                      # Multi-scale
                      if opt.multi_scale:
                          sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                          sf = sz / max(imgs.shape[2:])  # scale factor
                          if sf != 1:
                              ns = [math.ceil(x * sf / gs) * gs for x in
                                    imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                              imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                      # Forward
                      with amp.autocast(enabled=cuda):
                          pred = model(imgs)  # forward
                          loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                          if rank != -1:
                              loss *= opt.world_size  # gradient averaged between devices in DDP mode

                      # Backward
                      scaler.scale(loss).backward()

                      # Optimize
                      if ni % accumulate == 0:
                          scaler.step(optimizer)  # optimizer.step
                          scaler.update()
                          optimizer.zero_grad()
                          if ema:
                              ema.update(model)

                      # Print
                      if rank in [-1, 0]:
                          mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                          mem = '%.3gG' % (
                              torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                          s = ('%10s' * 2 + '%10.4g' * 6) % (
                              '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                          pbar.set_description(s)
                  # DDP process 0 or single-GPU
                  if rank in [-1, 0]:
                      # mAP
                      if ema:
                          ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                      final_epoch = epoch + 1 == epochs
                      if not opt.notest or final_epoch:  # Calculate mAP
                          results, maps, times = test.test(opt.data,
                                                           batch_size=total_batch_size,
                                                           imgsz=imgsz_test,
                                                           model=ema.ema,
                                                           single_cls=opt.single_cls,
                                                           dataloader=testloader,
                                                           save_dir=save_dir,
                                                           plots=plots and final_epoch,
                                                           log_imgs=opt.log_imgs if wandb else 0)

                      # Write
                      with open(results_file, 'a') as f:
                          f.write(
                              s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                      if len(opt.name) and opt.bucket:
                          os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

                      # Log
                      tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                              'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                              'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                              'x/lr0', 'x/lr1', 'x/lr2']  # params
                      for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                          if tb_writer:
                              tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                          if wandb:
                              wandb.log({tag: x})  # W&B
              if rank in [-1, 0]:
                  # Strip optimizers
                  n = opt.name if opt.name.isnumeric() else ''
                  fresults, flast, fbest = save_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
                  for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
                      if f1.exists():
                          os.rename(f1, f2)  # rename
                          if str(f2).endswith('.pt'):  # is *.pt
                              strip_optimizer(f2)  # strip optimizer
                              os.system(
                                  'gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
              # Train
              logger.info(opt)
              if not opt.evolve:
                  tb_writer = None  # init loggers
                  if opt.global_rank in [-1, 0]:
                      logger.info(
                          f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
                      tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
                  train(hyp, opt, device, tb_writer, wandb)

              # Evolve hyperparameters (optional)
              else:
                  # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
                  meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                          'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                          'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                          'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                          'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                          'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                          'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                          'box': (1, 0.02, 0.2),  # box loss gain
                          'cls': (1, 0.2, 4.0),  # cls loss gain
                          'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                          'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                          'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                          'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                          'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                          'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                          'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                          'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                          'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                          'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                          'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                          'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                          'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                          'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                          'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                          'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                          'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                          'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                          'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

                  assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
                  opt.notest, opt.nosave = True, True  # only test/save final epoch
                  # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
                  yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
                  if opt.bucket:
                      os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

                  for _ in range(300):  # generations to evolve
                      if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                          # Select parent(s)
                          parent = 'single'  # parent selection method: 'single' or 'weighted'
                          x = np.loadtxt('evolve.txt', ndmin=2)
                          n = min(5, len(x))  # number of previous results to consider
                          x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                          w = fitness(x) - fitness(x).min()  # weights
                          if parent == 'single' or len(x) == 1:
                              # x = x[random.randint(0, n - 1)]  # random selection
                              x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                          elif parent == 'weighted':
                              x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                          # Mutate
                          mp, s = 0.8, 0.2  # mutation probability, sigma
                          npr = np.random
                          npr.seed(int(time.time()))
                          g = np.array([x[0] for x in meta.values()])  # gains 0-1
                          ng = len(meta)
                          v = np.ones(ng)
                          while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                              v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                          for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                              hyp[k] = float(x[i + 7] * v[i])  # mutate

                      # Constrain to limits
                      for k, v in meta.items():
                          hyp[k] = max(hyp[k], v[1])  # lower limit
                          hyp[k] = min(hyp[k], v[2])  # upper limit
                          hyp[k] = round(hyp[k], 5)  # significant digits

                      # Train mutation
                      results = train(hyp.copy(), opt, device, wandb=wandb)

                      # Write mutation results
                      print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

                  # Plot results
                  plot_evolution(yaml_file)
                  print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                        f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

