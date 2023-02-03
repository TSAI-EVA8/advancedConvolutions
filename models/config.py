class Args:

    # Data Loading
    # ============

    train_batch_size = 64
    val_batch_size = 64
    num_workers = 4

    # Regularization
    # ==============
    dropout = 0.1

    # Training
    # ========
    random_seed = 1
    epochs = 10
    learning_rate = 0.01
    momentum = 0.9