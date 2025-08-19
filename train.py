from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

model = YOLO("yolo11n.yaml")



results = model.train(task="detect",data="yourdataset/data.yaml",device = [0,1],
                                                                              epochs=150,
                                                                              patience=150,
                                                                              batch=8,
                                                                              imgsz=640, 
                                                                              workers= 1,
                                                                              bgr= 0.0,
                                                                              box= 0.1887,
                                                                              cls= 0.27033,
                                                                              cutmix= 0.0,
                                                                              degrees= 20.0,
                                                                              fliplr= 0.5,
                                                                              flipud= 0.0,
                                                                              hsv_h= 0.0,
                                                                              hsv_s= 0.6872304839691561,
                                                                              hsv_v= 0.8889545777864033,
                                                                              lr0= 0.0010317,
                                                                              lrf= 0.64506,
                                                                              mixup= 0.0,
                                                                              momentum= 0.64506,
                                                                              mosaic= 0.0,
                                                                              name= "ShampooSTFA",
                                                                              optimizer= "AdamW",
                                                                              perspective= 1.6132948989127868e-05,
                                                                              scale= 0.5,
                                                                              shear= 10.0,
                                                                              translate= 0.1,
                                                                              warmup_epochs= 3.1,
                                                                              warmup_momentum= 0.45676,
                                                                              weight_decay= 0.0004008345021896791
                                                                                    ) 
