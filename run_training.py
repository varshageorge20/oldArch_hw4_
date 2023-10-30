from homework.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir")
    # Put custom arguments here
    # Put custom arguments here
    # putting all the custom arguments here with their default values
    # parser.add_argument('-k', '--change', default='Compose([ColorJitter(0.8, 0.8, 0.8, 0.2), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument("-lr", "--rate_gain", type=float, default=1e-4)
    parser.add_argument("-u", "--wt", type=float, default=0.02)
    parser.add_argument("-n", "--count", type=int, default=180)
    parser.add_argument("-ln", "--rate_ln", action="store_true")
    parser.add_argument("-sw", "--size_weight", default=0.5)

    args = parser.parse_args()
    train(args)
