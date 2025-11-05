import argparse
import training
import prediction_test

def main():
    parser = argparse.ArgumentParser(prog="cells", description="My simple CLI tool")
    parser.add_argument("command", choices=["train", "predict"], help="Run training or prediction")
    args = parser.parse_args()

    if args.command == "train":
        # get all defaults from training's parser
        train_args = training.parse_args([])   # empty list = no CLI options, just defaults

        # override only what you care about
        train_args.dataset_size = 6
        train_args.testing_size = 1
        train_args.epochs = [25]

        training.main(train_args)

    elif args.command == "predict":
        prediction_test.main()

if __name__ == "__main__":
    main()