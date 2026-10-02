"""Download the DSEC optical-flow benchmark's test split (7 recordings without ground truth).

Files are laid out as `DSECDataset(target_type="optical_flow_test")` expects:

    {data_path}/DSEC/test/{recording}/events_left/events.h5      (+ rectify_map.h5)
    {data_path}/DSEC/test/{recording}/test_forward_flow_timestamps.csv

    python scripts/download_dsec_test.py --data-path ~/data

Each event zip is several GB. Files already on disk are skipped, so it is safe to re-run.
"""
import argparse
import os
import shutil
import zipfile

from dell.datasets.download import download_file
from dell.datasets.raw.dsec import FLOW_TEST_RECORDINGS

TEST_SEQUENCES = FLOW_TEST_RECORDINGS
_BASE_TEST_URL = "https://download.ifi.uzh.ch/rpg/DSEC/test"
_TIMESTAMPS_ZIP_URL = "https://download.ifi.uzh.ch/rpg/DSEC/test_forward_optical_flow_timestamps.zip"


def _download_timestamps(dsec_test_root: str) -> None:
    """Download the timestamps zip and extract each recording's csv to its own folder."""
    zip_path = os.path.join(dsec_test_root, "_test_forward_flow_timestamps.zip")
    if not os.path.isfile(zip_path):
        download_file(_TIMESTAMPS_ZIP_URL, zip_path, "DSEC test flow timestamps")

    with zipfile.ZipFile(zip_path) as archive:
        for recording in TEST_SEQUENCES:
            destination = os.path.join(dsec_test_root, recording, "test_forward_flow_timestamps.csv")
            if os.path.isfile(destination):
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with archive.open(f"{recording}.csv") as source, open(destination, "wb") as target_file:
                shutil.copyfileobj(source, target_file)
    os.remove(zip_path)


def _download_events(dsec_test_root: str, recording: str) -> None:
    events_dir = os.path.join(dsec_test_root, recording, "events_left")
    if os.path.isdir(events_dir):
        print(f"[download_dsec_test] {events_dir} already exists, skipping")
        return

    zip_path = os.path.join(dsec_test_root, recording, "events_left.zip")
    url = f"{_BASE_TEST_URL}/{recording}/{recording}_events_left.zip"
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    download_file(url, zip_path, f"{recording} events_left")
    shutil.unpack_archive(zip_path, events_dir)
    os.remove(zip_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=os.path.expanduser("~/data"),
                        help="dataset root, as passed to run.py; DSEC/ is created inside it (default: ~/data)")
    parser.add_argument("--sequences", nargs="+", default=TEST_SEQUENCES, choices=TEST_SEQUENCES,
                        help="test sequences to download (default: all 7)")
    args = parser.parse_args(argv)

    dsec_test_root = os.path.join(args.data_path, "DSEC", "test")
    os.makedirs(dsec_test_root, exist_ok=True)

    _download_timestamps(dsec_test_root)
    for recording in args.sequences:
        _download_events(dsec_test_root, recording)

    print(f"[download_dsec_test] done -- test split ready under {dsec_test_root}")


if __name__ == "__main__":
    main()
