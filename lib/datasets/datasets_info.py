
def get_datasets_info(dataset, use_dict = False, test=False):
    datasets_info = dict()
    datasets_info['imdb_name'], datasets_info['imdbval_name'],  datasets_info['dataset'],  datasets_info['imdb_name'],  datasets_info['USE_FLIPPED'],  datasets_info['RPN_BATCHSIZE'],\
    datasets_info['BATCH_SIZE'],  datasets_info['RPN_POSITIVE_OVERLAP'],  datasets_info['RPN_NMS_THRESH'], \
    datasets_info['POOLING_SIZE_H'], datasets_info['POOLING_SIZE_W'], datasets_info['FG_THRESH'], datasets_info['SCALES'], datasets_info['sample_mode'],\
    datasets_info['VGG_ORIGIN'], datasets_info['USE_ALL_GT'], datasets_info['ignore_people'], datasets_info['filter_empty'], datasets_info['DEBUG'], datasets_info['set_cfgs'] \
           = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    if dataset == "pascal_voc":
        datasets_info['imdb_name'] = "voc_2007_trainval"
        datasets_info['imdbval_name'] = "voc_2007_test"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        datasets_info['MAX_NUM_GT_BOXES'] = 20
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['ANCHOR_SCALES'] = [8, 16, 32]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 21
        datasets_info['USE_FLIPPED'] = True
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['SCALES'] =(600,)
        datasets_info['dataset'] = dataset
        #set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        if test:
            datasets_info['imdb_name'] = "voc_2007_test"
    elif dataset == "pascal_voc_0712":
        datasets_info['imdb_name'] = "voc_2007_trainval+voc_2012_trainval"
        datasets_info['imdbval_name'] = "voc_2007_test"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        datasets_info['MAX_NUM_GT_BOXES'] = 20
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        # datasets_info['ANCHOR_SCALES'] = [4, 8, 16, 32]
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 21
        datasets_info['USE_FLIPPED'] = True 
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['SCALES'] =(600,)
        datasets_info['dataset'] = dataset
        if test:
            datasets_info['imdb_name'] = "voc_2007_test"
    elif dataset == "coco":
        datasets_info['imdb_name'] = "coco_2014_valminusminival"
        datasets_info['SCALES'] =(800,)
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['imdbval_name'] = "coco_2014_minival"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        datasets_info['MAX_NUM_GT_BOXES'] = 50
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 81
        datasets_info['dataset'] = dataset
        datasets_info['USE_FLIPPED'] = True
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
    elif dataset == "imagenet":
        datasets_info['imdb_name'] = "imagenet_train"
        datasets_info['imdbval_name'] = "imagenet_val"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['dataset'] = dataset
        datasets_info['MAX_NUM_GT_BOXES'] = 50 
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 4
        datasets_info['USE_FLIPPED'] = True
    elif dataset == "caltech":
        datasets_info['imdb_name'] = "caltech_train"
        datasets_info['imdbval_name'] = "caltech_val"
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 32
        datasets_info['dataset'] = dataset
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['FG_THRESH'] = 0.45
        if test:
            datasets_info['imdb_name'] = "caltech_test"
        if test:
            datasets_info['RPN_NMS_THRESH'] = 0.65
        else:
            datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['SCALES'] =(720,)
        datasets_info['MAX_NUM_GT_BOXES'] = 20 
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [2]
        ## scales*11 is the new_width, new_width*ratio is new height
        # 30, 42, 60, 84, 120, 168, 240
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
        datasets_info['num_classes'] = 4
        datasets_info['USE_FLIPPED'] = True 
    elif dataset == "KITTIVOC":
        datasets_info['imdb_name'] = "kittivoc_train"
        datasets_info['imdbval_name'] = "kittivoc_val"
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 128
        if test:
            datasets_info['imdb_name'] = "kittivoc_val"
        datasets_info['dataset'] = dataset
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['FG_THRESH'] = 0.45
        datasets_info['imdb_name'] = "kittivoc_train"
        datasets_info['SCALES']=(576,)
        datasets_info['MAX_NUM_GT_BOXES'] = 20
        ## scales*11 is the new_width, new_width*ratio is new height
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 4
        datasets_info['USE_FLIPPED'] = True
    elif dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        datasets_info['imdb_name'] = "vg_150-50-50_minitrain"
        datasets_info['imdbval_name'] = "vg_150-50-50_minival"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        datasets_info['MAX_NUM_GT_BOXES'] = 50
        datasets_info['ANCHOR_SCALES'] = [2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32] 
        datasets_info['ANCHOR_RATIOS'] = [1, 2]
        datasets_info['num_classes'] = 4
        datasets_info['USE_FLIPPED'] = True 
    elif dataset == "KAISTVOC":
        datasets_info['imdb_name'] = "kaist_train"
        datasets_info['imdbval_name'] = "kaist_test"
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = imdb_name
        datasets_info['USE_FLIPPED'] = True 
        datasets_info['RPN_BATCHSIZE'] = 128
        datasets_info['BATCH_SIZE'] =32
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['POOLING_SIZE_H'] = 8
        datasets_info['POOLING_SIZE_W'] = 4
        datasets_info['FG_THRESH'] = 0.45
        datasets_info['SCALES'] =(720,)
        datasets_info['sample_mode'] = 'bootstrap' # use bootstrap or ramdom as sampling method
        datasets_info['USE_ALL_GT'] = True # choose true if want to exclude all proposals overlap with 'people' larger than 0.3
        datasets_info['ignore_people'] = False # ignore people, all proposals overlap with 'people' larger than 0.3 will be igonored
        datasets_info['filter_empty'] = False # whether filter 0 gt images
        datasets_info['DEBUG'] = False # set as True if debug whether 'people' is ignored
        ## scales*11 is the new_width, new_width*ratio is new height
        # 30, 42, 60, 84, 120, 168, 240 width
        datasets_info['MAX_NUM_GT_BOXES'] = 20
        datasets_info['ANCHOR_SCALES'] = [2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32] 
        datasets_info['ANCHOR_RATIOS'] = [1, 2]
        datasets_info['num_classes'] = 4
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[1, 2]', 'MAX_NUM_GT_BOXES', '20']
    elif dataset == 'widerface':
        datasets_info['imdbval_name'] = "widerface_val"
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = "widerface_train"
        datasets_info['SCALES'] = (800,)
        datasets_info['MAX_NUM_GT_BOXES'] = 300
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        if test:
            datasets_info['RPN_NMS_THRESH'] = 0.7
        else:
            datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['FG_THRESH'] = 0.45
        datasets_info['sample_mode'] = 'random'
        datasets_info['filter_empty'] = False
        datasets_info['DEBUG'] = False
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[1]', 'MAX_NUM_GT_BOXES', '300']
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [1]
        datasets_info['num_classes'] = 2
        datasets_info['USE_FLIPPED'] = True
    elif dataset in ["clipart","comic","watercolor",'cross_domain']:
        if dataset == "clipart" :
            datasets_info['imdb_name'] = "clipart_train"
            datasets_info['imdbval_name'] = "clipart_test"
        elif dataset == "comic" :
            datasets_info['imdb_name'] = "comic_train"
            datasets_info['imdbval_name'] = "comic_test"
        elif dataset == "watercolor" :
            datasets_info['imdb_name'] = "watercolor_train"
            datasets_info['imdbval_name'] = "watercolor_test"
        elif dataset == 'cross_domain':
            datasets_info['imdb_name'] = "watercolor_train+clipart_train+comic_train"
            datasets_info['imdbval_name'] = "watercolor_test"  
        datasets_info['dataset'] = dataset
        datasets_info['SCALES'] = (600,) # origin is 600
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.3
        datasets_info['USE_FLIPPED'] = True
        datasets_info['filter_empty'] = True
        datasets_info['sample_mode'] = 'random'
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['MAX_NUM_GT_BOXES'] = 30
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 21
        datasets_info['set_cfgs']  = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']
    elif dataset == "citypersons":
        datasets_info['imdb_name'] = "citypersons_train"
        datasets_info['imdbval_name'] = "citypersons_val"
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[2.72, 3.81, 5.45, 7.64, 10.9, 15.27, 21.8, 32]', 'ANCHOR_RATIOS', '[2]', 'MAX_NUM_GT_BOXES', '20']
        datasets_info['MAX_NUM_GT_BOXES'] = 20
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['ANCHOR_SCALES'] = [8, 16, 32]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 21
        datasets_info['USE_FLIPPED'] = True
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['SCALES'] =(600,)
        datasets_info['dataset'] = dataset
    elif dataset == "dota":
        datasets_info['imdbval_name'] = "dota_val"
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = "dota_train"
        datasets_info['SCALES'] = (600,)
        datasets_info['MAX_NUM_GT_BOXES'] = 100
        datasets_info['RPN_BATCHSIZE'] = 128
        datasets_info['BATCH_SIZE'] = 128
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        if test:
            datasets_info['RPN_NMS_THRESH'] = 0.65
        else:
            datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['sample_mode'] = 'random'
        datasets_info['filter_empty'] = False
        datasets_info['DEBUG'] = False
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '100']
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 16
        datasets_info['USE_FLIPPED'] = True 

    elif dataset == "Kitchen":
        datasets_info['imdbval_name'] = "kitchen_test"
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = "kitchen_train"
        datasets_info['SCALES'] = (800,) # origin is 1024
        datasets_info['MAX_NUM_GT_BOXES'] = 30
        datasets_info['RPN_BATCHSIZE'] = 256
        datasets_info['BATCH_SIZE'] = 256
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.7
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['sample_mode'] = 'random'
        datasets_info['filter_empty'] = False
        datasets_info['DEBUG'] = False
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '30']
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 12
        datasets_info['USE_FLIPPED'] = True
    
    elif dataset == "deeplesion":
        datasets_info['imdbval_name'] = "deeplesion_test"
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = "deeplesion_trainval"
        datasets_info['SCALES'] = (512,) # origin is 1024
        datasets_info['MAX_NUM_GT_BOXES'] = 30
        datasets_info['RPN_BATCHSIZE'] = 128
        datasets_info['BATCH_SIZE'] = 64
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.7
        datasets_info['FG_THRESH'] = 0.5
        datasets_info['sample_mode'] = 'random'
        datasets_info['filter_empty'] = False
        datasets_info['DEBUG'] = False
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '10']
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 2
        datasets_info['USE_FLIPPED'] = False

    elif dataset == "LISA":
        datasets_info['imdbval_name'] = "LISA_test"
        datasets_info['POOLING_SIZE_H'] = 7
        datasets_info['POOLING_SIZE_W'] = 7
        datasets_info['dataset'] = dataset
        datasets_info['imdb_name'] = "LISA_train"
        datasets_info['SCALES'] = (800,) # origin is 1024
        datasets_info['MAX_NUM_GT_BOXES'] = 30
        datasets_info['RPN_BATCHSIZE'] = 64
        datasets_info['BATCH_SIZE'] = 32
        datasets_info['RPN_POSITIVE_OVERLAP'] = 0.5
        datasets_info['RPN_NMS_THRESH'] = 0.65
        datasets_info['FG_THRESH'] = 0.45
        datasets_info['sample_mode'] = 'random'
        datasets_info['filter_empty'] = False
        datasets_info['DEBUG'] = False
        datasets_info['set_cfgs'] = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '10']
        # datasets_info['ANCHOR_SCALES'] = [4, 8, 16, 32]
        datasets_info['ANCHOR_SCALES'] = [0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 30]
        datasets_info['ANCHOR_RATIOS'] = [0.5, 1, 2]
        datasets_info['num_classes'] = 5
        datasets_info['USE_FLIPPED'] = True
        # cfg.TRAIN.BATCH_SIZE = 256
        # cfg.TRAIN.RPN_BATCHSIZE = 256


    if use_dict:
        return datasets_info
    else:
        return  datasets_info['imdb_name'], datasets_info['imdbval_name'],  datasets_info['dataset'],  datasets_info['imdb_name'],  datasets_info['USE_FLIPPED'],  datasets_info['RPN_BATCHSIZE'],\
                datasets_info['BATCH_SIZE'],  datasets_info['RPN_POSITIVE_OVERLAP'],  datasets_info['RPN_NMS_THRESH'], \
                datasets_info['POOLING_SIZE_H'], datasets_info['POOLING_SIZE_W'], datasets_info['FG_THRESH'], datasets_info['SCALES'], datasets_info['sample_mode'],\
                datasets_info['VGG_ORIGIN'], datasets_info['USE_ALL_GT'], datasets_info['ignore_people'], datasets_info['filter_empty'], datasets_info['DEBUG'], datasets_info['set_cfgs']
    
def univ_info(datasets_list, variable, test=False):
    # print(datasets_list)
    # print(variable)
    list = [get_datasets_info(datasets, use_dict=True, test=test)[variable] for datasets in datasets_list]
    return list