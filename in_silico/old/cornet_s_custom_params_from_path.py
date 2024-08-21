def cornet_s_custom_params_from_path(path):

    # break down path and find start of useful info
    model_info = path.split('/')
    start = model_info.index('data')

    # model name
    model_name = model_info[start+1]

    # model
    model_arch = model_info[start+2].split('_')
    Rstr, Kstr, Fstr = model_arch[:3]
    R = [int(x) for x in Rstr.split('-')[1:]]
    K = [int(x) for x in Kstr.split('-')[1:]]
    F = [int(x) for x in Fstr.split('-')[1:]]
    S = int(model_arch[3].split('-')[-1])

    # dataset
    dataset = model_info[start+3].split('_')[0]
    occluder_train = model_info[start+3][(len(dataset)+1):]

    # training
    learning = model_info[start+4]
    identifier = model_info[start+5]

    return model_name, Rstr, R, Kstr, K, Fstr, F, S, dataset, occluder_train, learning, identifier