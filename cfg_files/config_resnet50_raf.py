from base_config import Config

base_config = Config({
    'name': 'Base Config',

    'backbone': 'resnet50',
    'feature_dim': 512,
    'pretrained': 'msceleb',
    
    'input_size': [112, 112],
    'pad_size': 4,
    'batch_size': 32,
    'num_parallel_calls': -1,  # number of workers for processing data

    'optimizer': 'adam',  # adam or sgd
    'lr': 1e-4,  # initial learning rate
    'lr_decay': 0.1,
    'lr_steps': [10, 30],  # after some interval (epochs), decay the lr by multiplying with lr_decay


    'gamma': 0.01,
    'num_neighbors': 8,
    'lamb_init': 0.5,
    'lamb_lr': 10,
    'lamb_beta': 0,

    'num_classes': 7,
    'class_names': ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"],
    'class_weights': None,
    'val_interval': 200,  # validate after number of iterations
    'save_interval': 2,  # save model after number of epochs
    'epochs': 60,
    'checkpoint_dir': "weights_checkpoint/resnet50_raf"
})

config = base_config.copy()
