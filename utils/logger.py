import os
import csv


class MetricLogger:
    def __init__(self, save_dir, filename="train_log.csv"):
        self.save_dir = save_dir
        self.filename = filename
        self.log_path = os.path.join(save_dir, filename)

        os.makedirs(save_dir, exist_ok=True)

        self.header_written = False

    def write(self, row_dict):

        file_exists = os.path.exists(self.log_path)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))

            if not file_exists:
                writer.writeheader()

            writer.writerow(row_dict)