import torch
import numpy as np

from homework.models import Detector, save_model
from homework.utils import load_detection_data
from homework import dense_transforms
import torch.utils.tensorboard as tb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = Detector().to(device)

change = dense_transforms.Compose(
    [
        dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        dense_transforms.RandomHorizontalFlip(0.5),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap(),
    ]
)


def train(args):
    from os import path

    model = Detector()
    model = model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    global_step = 0  # keeps track of the # of training steps
    torch.save(
        model.state_dict(), (path.join(path.dirname(path.abspath(__file__)), "det.th"))
    )

    if args.rate_ln:
        print("hello")
        # loads the model's parameters from saved checkpoint file called det.th.
        model.load_state_dict(
            torch.load(path.join(path.dirname(path.abspath(__file__)), "det.th"))
        )
        # evaluating expression; constructs dictionary which maps class names of dense_tranforms.py to corresponding class objects
        # change = eval(args.change, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

        # loading data that may undergo transformations
        train_data = load_detection_data(
            "dense_data/train", transform=change, batch_size=256
        )
        validation_data = load_detection_data(
            "dense_data/valid", transform=change, batch_size=256
        )

        # MSE loss will be used during training. reduction=none means it won't be reduced to a scalar value
        mean_squared_error_loss = torch.nn.MSELoss(reduction="mean")

        # binary cross-entropy (BCE) loss function with logits
        binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # optimizer=Adam, The model's parameters are updated using this optimizer during training
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.rate_gain, weight_decay=1e-4
        )

        for epoch in range(args.count):
            print(f"Starting Epoch: {epoch}")

            # img = image, dg=ground truth detection maps, sg=other data. these batches are provided by data loader. dt=training data
            training_peak_center_loss_per_batch = []
            for i, (image, peak_center_ground_truth, size_ground_truth) in enumerate(
                train_data
            ):
                print(f"Loading training batch: {i}", end="\r")
                image, peak_center_ground_truth, size_ground_truth = (
                    image.to(device),
                    peak_center_ground_truth.to(device),
                    size_ground_truth.to(device),
                )

                peak_detections, size_detections = model(image)

                validation_peak_detection_loss = binary_cross_entropy_loss(
                    peak_detections, peak_center_ground_truth
                )

                total_loss = validation_peak_detection_loss
                training_peak_center_loss_per_batch.append(total_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # checking for logging
                if train_logger is not None and global_step % 5 == 0:
                    log(
                        train_logger,
                        image,
                        peak_center_ground_truth,
                        peak_detections,
                        global_step,
                    )
            print()

            validation_peak_center_loss_per_batch = []
            for i, (image, peak_center_ground_truth, size_ground_truth) in enumerate(
                validation_data
            ):
                print(f"Loading validation batch: {i}", end="\r")

                image, peak_center_ground_truth, size_ground_truth = (
                    image.to(device),
                    peak_center_ground_truth.to(device),
                    size_ground_truth.to(device),
                )

                peak_detections, size_detections = model(image)

                validation_peak_detection_loss = binary_cross_entropy_loss(
                    peak_detections, peak_center_ground_truth
                ).detach()

                total_loss = validation_peak_detection_loss
                validation_peak_center_loss_per_batch.append(total_loss)

                # checking for logging
                if valid_logger is not None and global_step % 5 == 0:
                    log(
                        valid_logger,
                        image,
                        peak_center_ground_truth,
                        peak_detections,
                        global_step,
                    )
            print()

            train_loss = sum(training_peak_center_loss_per_batch) / len(
                training_peak_center_loss_per_batch
            )
            validation_loss = sum(validation_peak_center_loss_per_batch) / len(
                validation_peak_center_loss_per_batch
            )
            print(f"Training loss: {train_loss}, validation loss: {validation_loss}")

            if train_logger is not None:
                train_logger.add_scalar("ld", train_loss, global_step)
            if valid_logger is not None:
                valid_logger.add_scalar("ld", validation_loss, global_step)

            global_step = global_step + 1
            save_model(model)

    # raise NotImplementedError('train')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images("image", imgs[:16], global_step)
    logger.add_images("label", gt_det[:16], global_step)
    logger.add_images("pred", torch.sigmoid(det[:16]), global_step)


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
