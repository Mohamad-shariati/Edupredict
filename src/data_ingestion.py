"""
This module handles downloading raw CSV data from a specified URL,
splitting it into train/validation/test sets, and saving the data as CSV files.
"""

import random
from pathlib import Path
from urllib.request import urlopen

from config_reader import read_config
from logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.data_ingestion_config = config["data_ingestion"]
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.object_name = self.data_ingestion_config["object_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]
        self.train_ratio = self.data_ingestion_config["train_ratio"]
        self.val_ratio = self.data_ingestion_config["val_ratio"]
        self.url = f"https://{self.bucket_name}.{self.storage_path}/{self.object_name}"

        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.raw_dir = artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self):
        """
        Download raw CSV data from the configured URL.

        Returns
        -------
        str
            Raw CSV data as a string.
        """
        try:
            logger.info(f"Downloading data from {self.url} ...")
            with urlopen(self.url) as response:
                raw_data = response.read().decode("utf-8")
            logger.info("Download successful.")
            return raw_data
        except Exception as e:
            logger.error(f"Failed to download data from {self.url}: {str(e)}")
            raise

    def split_data(self, raw_data_str):
        """
        Split CSV data (with header) into train, validation, and test sets.

        Parameters
        ----------
        raw_data_str : str
            Raw CSV data as a string.

        Returns
        -------
        tuple
            (header, train_data, val_data, test_data)
        """
        lines = raw_data_str.strip().split("\n")
        header = lines[0]
        rows = lines[1:]

        random.shuffle(rows)

        total = len(rows)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_data = rows[:train_end]
        val_data = rows[train_end:val_end]
        test_data = rows[val_end:]

        logger.info(f"Split summary:")
        logger.info(f"Train set: {len(train_data)} records")
        logger.info(f"Validation set: {len(val_data)} records")
        logger.info(f"Test set: {len(test_data)} records")

        return header, train_data, val_data, test_data

    def save_to_csv_files(self, header, train_data, val_data, test_data):
        """
        Save the split data into CSV files.

        Parameters
        ----------
        header : str
            CSV header line.
        train_data : list
            Training data rows.
        val_data : list
            Validation data rows.
        test_data : list
            Test data rows.
        """
        data_files = [
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ]

        for name, data in data_files:
            output_file = self.raw_dir / f"{name}.csv"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(header)
                for row in data:
                    f.write(row)
            logger.info(f"Saved {name} data with {len(data)} records to {output_file}")

    def run(self):
        """
        Execute the complete data ingestion pipeline.

        This method orchestrates the entire data ingestion process:
        1. Downloads raw data
        2. Splits it into train/validation/test sets
        3. Saves the processed data as CSV files

        Examples
        --------
        >>> data_ingestion = DataIngestion(read_config("config/config.yaml"))
        >>> data_ingestion.run()
        """
        logger.info(f"Data Ingestion started for {self.url}")
        raw_data = self.download_raw_data()
        header, train_data, val_data, test_data = self.split_data(raw_data)
        self.save_to_csv_files(header, train_data, val_data, test_data)
        logger.info("Data Ingestion completed successfully.")
