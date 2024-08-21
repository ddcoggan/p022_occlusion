from argparse import Namespace

def cornet_s_custom_params_from_path(path):

    # break down path and find start of useful info
    model_name, identifier = path.split('/')[2:4]

    class CFG:

        M = Namespace(
            model_name=model_name,
            identifier=identifier,  # used to name model directory, required
        )


        # cornet_s_custom default parameters
        if M.model_name == 'cornet_s_custom':                             # used to load architecture, required
            M.R = (1,2,4,2)                                       # recurrence, default = (1,2,4,2),
            M.K = (3,3,3,3)                                       # kernel size, default = (3,3,3,3),
            M.F = (64,128,256,512)                                # feature maps, default = (64,128,256,512)
            M.S = 4                                               # feature maps scaling, default = 4
            M.out_channels = 1                                    # number of heads, default = 1
            M.head_depth = 1                                     # multi-layer head, default = 1

        # cornet_st/flab default parameters
        elif M.model_name in ['cornet_flab', 'cornet_st']:
            M.model_name = 'cornet_flab'  # used to load architecture, required
            M.kernel_size = (3, 3, 3, 3)                          # kernel size, default = (3,3,3,3),
            M.num_features = (64,128,256,512)                     # feature maps, default = (64,128,256,512)
            M.times = 2
            M.out_channels = 1  # number of heads, default = 1
            M.head_depth = 1  # multi-layer head, default = 1

        # dataset
        D = Namespace(
            dataset='ILSVRC2012',
            contrast='occluder_translate',  # contrastive learning manipulation. Options: 'repeat_transform','occluder_translate'
        )

        T = Namespace(learning='supervised_classification')

    # change architecture params based on identifier
    if 'kern-5' in identifier:
        CFG.M.K = (5,5,5,5)
    if 'kern-7' in identifier:
        CFG.M.K = (7,7,7,7)
    if 'feat-512' in identifier:
        CFG.M.F = (512,512,512,512)
    if 'rec-0' in identifier:
        CFG.M.R = (1,1,1,1)
    if 'rec-2xV1' in identifier:
        CFG.M.R = (2,2,4,2)
    elif 'rec2x' in identifier:
        CFG.M.R = (2,4,8,4)
    if 'head-deep' in identifier:
        CFG.M.head_depth = 2
    if 'task-class-cont' in identifier:
        CFG.M.out_channels = 2

    return CFG