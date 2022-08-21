from base_config import Config

base_config = Config({
    'name': 'Config 2',

    'backbone': 'resnet18',
    'feature_dim': 512,
    'pretrained': 'msceleb',

    'input_size': [224, 224],
    'pad_size': 8,
    'batch_size': 32,
    'num_parallel_calls': -1,  # number of workers for processing data

    'optimizer': 'adam',  # adam or sgd

    'lr': 1e-3,  # initial learning rate
    'lr_decay': 0.1,
    'lr_steps': [10, 20, 30],  # after some interval (epochs), decay the lr by multiplying with lr_decay


    'gamma': 0.1,
    'num_neighbors': 8,

    'num_classes': 7,
    'class_names': ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"],

    'val_interval': 1,  # validate after number of epochs
    'save_interval': 2,  # save model after number of epochs
    'epochs': 60,
    'checkpoint_dir': "weights_checkpoint/resnet18"
})

config = base_config.copy()
