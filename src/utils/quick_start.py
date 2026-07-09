# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))


def eval_only(model, dataset, config_dict, ckpt=None, eval_modes=None):
    """Train-once-eval-many: load a CLEAN-trained checkpoint and evaluate it
    under multiple MQS modes on the TEST set. No training, no early stopping
    contamination — the same model is scored under every quality shift.

    eval_modes: list of robust_eval_mode strings (e.g. ['normal','mismatch']).
    """
    from utils.misc import set_random_seed
    config = Config(model, dataset, config_dict, mg=False)
    init_logger(config)
    logger = getLogger()
    logger.info('██Eval-only MQS scan██')
    logger.info('██Dir: \t' + os.getcwd() + '\n')

    ds = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = ds.split()
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset,
                               batch_size=config['eval_batch_size'])

    if eval_modes is None:
        eval_modes = ['normal']

    # build model with a fixed seed (deterministic init, then overwritten by ckpt)
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    seed = config['seed']
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
    set_random_seed(seed)
    mdl = get_model(config['model'])(config, train_data).to(config['device'])

    if ckpt:
        state = torch.load(ckpt, map_location=config['device'])
        mdl.load_state_dict(state)
        logger.info('Loaded checkpoint: ' + ckpt)
    else:
        logger.info('[WARN] no checkpoint given; evaluating random init (debug only)')

    trainer = get_trainer()(config, mdl, mg=False)

    logger.info('\n=================================\nMQS eval modes: {}\n'.format(eval_modes))
    for mode in eval_modes:
        mdl.robust_eval_mode = mode
        # noise std / shift ratio / tail ratio already set from config; keep them
        result = trainer.evaluate(test_data, is_test=True)
        logger.info('>>>>> eval_mode={} | {}\n'.format(mode, dict2str(result)))
    logger.info('\n============Eval-only Over=====================\n')

