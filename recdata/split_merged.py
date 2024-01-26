import sys
sys.path.append("../pymodules")
from custom_read_csv import KeyPressSequence


def export_session_kps(kps: KeyPressSequence,
                       filepath: str | None = None) -> None:
    """Given a KeyPressSequence, export it to a .csv using the "newer format",
    with columns: key_ascii,touch,hit,release,session

    If filepath is set to `None`, the file is created in the current directory,
    and its name is set to the session of the first KeyPressData in `kps.data`
    """
    if filepath is None:
        filepath = f"kps_{kps.data[0].session}.csv"
    file = open(filepath, "w")
    file.write("key_ascii,touch,hit,release,session\n")
    for kpd in kps.data:
        key_ascii = ord(kpd.key)
        t = " ".join(map(str, kpd.touch.tolist()))
        h = " ".join(map(str, kpd.hit.tolist()))
        r = " ".join(map(str, kpd.release.tolist()))
        file.write(f"{key_ascii},{t},{h},{r},{kpd.session}\n")
    return None


if __name__ == "__main__":
    MERGED_FILE_PATH = None  # MUST BE SPECIFIED HERE
    TRAIN_FILE_PATH = None  # optional to specify, see export_session_kps() for default
    TEST_FILE_PATH = None  # optional to specify, see export_session_kps() for default
    main_kps = KeyPressSequence(MERGED_FILE_PATH)
    train_kps = KeyPressSequence()
    test_kps = KeyPressSequence()
    for kpd in main_kps.data:
        if kpd.session[:4] == "test":
            test_kps.data.append(kpd)
        else:
            train_kps.data.append(kpd)
    export_session_kps(train_kps, TRAIN_FILE_PATH)
    export_session_kps(test_kps, TEST_FILE_PATH)
