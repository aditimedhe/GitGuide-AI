import hashlib
import json
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRACKER_DIR = PROJECT_ROOT / "vectorstore"

TRACKER_FILE = TRACKER_DIR / "processed_documents.json"


# --------------------------------------------------
# Create tracker file if required
# --------------------------------------------------

def initialize_tracker():
    """
    Create the tracker directory and JSON file
    if they do not already exist.
    """

    TRACKER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not TRACKER_FILE.exists():

        TRACKER_FILE.write_text(
            "{}",
            encoding="utf-8"
        )


# --------------------------------------------------
# Calculate file hash
# --------------------------------------------------

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.

    The hash allows us to detect whether a document
    has changed.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# --------------------------------------------------
# Load processing history
# --------------------------------------------------

def load_tracker():
    """
    Load processed document information.
    """

    initialize_tracker()

    try:

        data = json.loads(
            TRACKER_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

        return {}

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {}


# --------------------------------------------------
# Save processing history
# --------------------------------------------------

def save_tracker(data):
    """
    Save processed document information.
    """

    initialize_tracker()

    TRACKER_FILE.write_text(
        json.dumps(
            data,
            indent=4
        ),
        encoding="utf-8"
    )


# --------------------------------------------------
# Check whether document needs processing
# --------------------------------------------------

def needs_processing(file_path):
    """
    Return True if:

    1. The document has never been processed.
    2. The document has changed since processing.

    Return False if the exact same file has already
    been processed.
    """

    tracker = load_tracker()

    file_path = Path(file_path)

    file_key = str(
        file_path.resolve()
    )

    current_hash = calculate_file_hash(
        file_path
    )

    existing_record = tracker.get(
        file_key
    )

    if existing_record is None:

        return True

    previous_hash = existing_record.get(
        "hash"
    )

    return previous_hash != current_hash


# --------------------------------------------------
# Mark document as processed
# --------------------------------------------------

def mark_as_processed(file_path):
    """
    Save the current hash of a successfully
    processed document.
    """

    tracker = load_tracker()

    file_path = Path(file_path)

    file_key = str(
        file_path.resolve()
    )

    file_hash = calculate_file_hash(
        file_path
    )

    tracker[file_key] = {
        "filename": file_path.name,
        "hash": file_hash,
    }

    save_tracker(tracker)


# --------------------------------------------------
# Get processing status
# --------------------------------------------------

def get_document_status(file_path):
    """
    Return information about whether a document
    has already been processed.
    """

    tracker = load_tracker()

    file_path = Path(file_path)

    file_key = str(
        file_path.resolve()
    )

    record = tracker.get(
        file_key
    )

    if record is None:

        return {
            "processed": False,
            "changed": False,
        }

    current_hash = calculate_file_hash(
        file_path
    )

    previous_hash = record.get(
        "hash"
    )

    return {
        "processed": True,
        "changed": current_hash != previous_hash,
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "\nDocument tracker initialized."
    )

    print(
        f"Tracker file: {TRACKER_FILE}"
    )