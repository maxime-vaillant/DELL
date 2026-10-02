import os
import time
import urllib.request
import warnings

from tqdm import tqdm

MAX_DOWNLOAD_ATTEMPTS = 5


def download_file(url: str, local_path: str, description: str) -> None:
    """Stream `url` to `local_path`, which only appears once the download is complete.

    The data goes to a `.part` file first, so an interrupted download (including Ctrl-C or a killed
    job) never leaves a truncated file that would later be treated as complete.
    """
    partial_path = local_path + ".part"
    try:
        with urllib.request.urlopen(url) as response:
            expected_size = int(response.headers.get("Content-Length", 0))
            with tqdm(total=expected_size, unit="B", unit_scale=True, desc=description) as progress_bar, \
                    open(partial_path, "wb") as partial_file:
                downloaded_size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    partial_file.write(chunk)
                    downloaded_size += len(chunk)
                    progress_bar.update(len(chunk))
        if expected_size and downloaded_size != expected_size:
            raise IOError(f"incomplete download of {url}: got {downloaded_size} of {expected_size} bytes")
        os.replace(partial_path, local_path)
    finally:
        if os.path.isfile(partial_path):
            os.remove(partial_path)


def download_file_with_retries(
    url: str, local_path: str, description: str, max_attempts: int = MAX_DOWNLOAD_ATTEMPTS,
) -> None:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            attempt_description = description + (f" (attempt {attempt})" if attempt > 1 else "")
            download_file(url, local_path, attempt_description)
            return
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                warnings.warn(
                    f"download of {url} failed on attempt {attempt}/{max_attempts} ({error!r}); retrying.",
                    stacklevel=3,
                )
                time.sleep(min(attempt * 5, 30))

    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts.") from last_error
