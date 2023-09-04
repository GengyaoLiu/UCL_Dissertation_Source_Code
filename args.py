import argparse


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--freeze',
                        type=str,
                        default="True",
                        help='Whether to freeze the pre-trained model in training')
    parser.add_argument('--augment',
                        type=str,
                        default="False",
                        help='Whether to augment the training and validate data set')
    parser.add_argument('--training',
                        type=str,
                        default="True",
                        help='Whether to train the whole network')
    parser.add_argument('--attention',
                        type=str,
                        default="False",
                        help='Whether to use the attention at the end')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--model_name',
                        type=str,
                        default="VIT_patch16",
                        choices=("VIT_patch16", "VIT_patch32", "ResNet_50", "ResNet_34", "ResVit"),
                        help='Choose the model to train and test.')
    parser.add_argument('--loss_name',
                        type=str,
                        default="Cross_Entropy",
                        choices=("Hybrid_loss", "Cross_Entropy"),
                        help='Choose the model to train and test.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='/content/gdrive/MyDrive/Dissertation/weights',
                        help='Base directory for saving information.')

    args = parser.parse_args()

    return args


