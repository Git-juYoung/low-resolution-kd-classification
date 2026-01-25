teacher_config = {
    'freeze_until' : 'layer2',

    'batch_size' : 64,
    'num_workers' : 2,
    'epochs' : 100,

    'lr': 1e-4,
    'weight_decay': 1e-4,

    'label_smoothing' : 0.0,

    'scheduler_factor': 0.1,
    'scheduler_patience': 5,
    'scheduler_threshold' : 1e-4,

    'early_stopping_patience': 12,
}

student_config = {
    'freeze_until' : 'layer2',

    'batch_size' : 64,
    'num_workers' : 2,
    'epochs' : 50,

    'lr': 3e-4,
    'weight_decay': 1e-4,

    'label_smoothing' : 0.1,

    'scheduler_factor': 0.1,
    'scheduler_patience': 3,
    'scheduler_threshold' : 1e-4,

    'early_stopping_patience': 7,
    
    'kd_temperature': 4.0,
    'kd_alpha': 0.7,
    'kd_beta': 0.3
}

