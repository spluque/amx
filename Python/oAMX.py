# pylint: disable=too-many-locals,invalid-name,no-member

"""Read binary AMX data file from Loggerhead Instruments older tag model

"""

import argparse
import numpy as np


_STR_MAX = 8
_NSENSORS = 12


_sensor_dtype = np.dtype([("chipName", "S" + str(_STR_MAX)),
                          ("nChan", "uint16"),
                          ("name", object, _NSENSORS),
                          ("units", object, _NSENSORS),
                          ("cal", np.float32, _NSENSORS)])
SID_spec_dtype = np.dtype([("SID", "S" + str(_STR_MAX)),
                           ("sidType", "uint16"),
                           ("nSamples", "uint32"),
                           ("sensor", _sensor_dtype),
                           ("DForm", "uint32"),
                           ("srate", "float32")])
SID_record_dtype = np.dtype([("SID_id", "uint32"),
                             ("data", object)])


def oAMX(file_name):
    """Read AMX binary file and parse components

    Parameters
    ----------
    file_name : str
        Path to configuration file.

    Returns
    -------
    tuple
        Tuple with (index, name in brackets):

        numpy.ndarray [0]
            header information array.
        numpy.ndarray [1]
            specification structure for data.
        numpy.ndarray [2]
            structured array with data and specifier.

    """
    with open(file_name, "rb") as fname:
        head_dtype = np.dtype([("Version", "uint32"),
                               ("UserID", "uint32"),
                               ("Voltage", "float32"),
                               ("sec", "uint8"),
                               ("min", "uint8"),
                               ("hour", "uint8"),
                               ("day", "uint8"),
                               ("month", "uint8"),
                               ("year", "int16"),
                               ("tzOffset", "int16")])
        df_head = np.zeros(1, dtype=head_dtype)
        df_head["Version"] = np.fromfile(fname, "uint32", 1)
        df_head["UserID"] = np.fromfile(fname, "uint32", 1)
        df_head["Voltage"] = np.fromfile(fname, "float32", 1)
        df_head["sec"] = np.fromfile(fname, "uint8", 1)
        df_head["min"] = np.fromfile(fname, "uint8", 1)
        df_head["hour"] = np.fromfile(fname, "uint8", 1)
        df_head["day"] = np.fromfile(fname, "uint8", 1)
        df_head["month"] = np.fromfile(fname, "uint8", 1)
        np.fromfile(fname, "uint8", 3)  # NU
        df_head["year"] = np.fromfile(fname, "int16", 1)
        df_head["tzOffset"] = np.fromfile(fname, "int16", 1)

        # Read in SID_SPECS until get all zeroes
        not_done = 1
        SID_spec = np.empty(0, dtype=SID_spec_dtype)

        while (not_done):
            sid_spec = np.empty(1, dtype=SID_spec_dtype)
            SID = np.fromfile(fname, np.uint8, _STR_MAX)
            sid_spec["SID"] = "".join([chr(i) for i in SID if i > 0])
            sidType = np.asscalar(np.fromfile(fname, "uint16", 1))
            sid_spec["sidType"] = sidType
            np.fromfile(fname, np.uint16, 1)  # NU
            nSamples = np.asscalar(np.fromfile(fname, "uint32", 1))
            sid_spec["nSamples"] = nSamples
            chip_name = np.fromfile(fname, "uint8", _STR_MAX)
            chipName = "".join([chr(i) for i in chip_name if i > 0])
            sid_spec["sensor"]["chipName"] = chipName
            nChan = np.asscalar(np.fromfile(fname, "uint16", 1))
            sid_spec["sensor"]["nChan"] = nChan
            np.fromfile(fname, "uint16", 1)  # NU
            for i in range(_NSENSORS):
                new_name = np.fromfile(fname, "uint8", _STR_MAX)
                new_name = "".join([chr(j) for j in new_name if j > 0])
                if not new_name:
                    continue
                sid_spec["sensor"]["name"][0, i] = new_name
            for i in range(_NSENSORS):
                new_unit = np.fromfile(fname, "uint8", _STR_MAX)
                new_unit = "".join([chr(j) for j in new_unit if j > 0])
                if not new_unit:
                    continue
                sid_spec["sensor"]["units"][0, i] = new_unit
            for i in range(_NSENSORS):
                new_cal = np.fromfile(fname, "float32", 1)
                if not new_cal:
                    continue
                sid_spec["sensor"]["cal"][0, i] = new_cal
            sid_spec["DForm"] = np.fromfile(fname, "uint32", 1)
            sid_spec["srate"] = np.fromfile(fname, "float32", 1)

            SID_spec = np.append(SID_spec, sid_spec, axis=0)

            if not sid_spec["nSamples"]:
                not_done = 0

        SID_spec = np.delete(SID_spec, -1, 0)  # delete last zeros

        # Read in next SID_REC header and data
        SID_rec = np.empty(0, dtype=SID_record_dtype)

        while True:
            sid_rec = np.empty(1, dtype=SID_record_dtype)

            # Make sure we haven't reached EOF
            try:
                sid_rec["SID_id"] = np.fromfile(fname, "uint32", 1)
            except ValueError as error:
                break

            np.fromfile(fname, "uint32", 1)  # NU
            np.fromfile(fname, "uint32", 1)  # NU
            np.fromfile(fname, "uint32", 1)  # NU
            sid_id = sid_rec["SID_id"]

            if sid_id in range(7):
                sid = np.asscalar(SID_spec[sid_id]["SID"])
                nSamples = np.asscalar(SID_spec[sid_id]["nSamples"])
                DForm = np.asscalar(SID_spec[sid_id]["DForm"])
                if DForm == 2:
                    if sid.startswith("I") or sid.startswith("3"):
                        sid_rec["data"] = [np.fromfile(fname, ">i2",
                                                       nSamples)]
                    else:
                        sid_rec["data"] = [np.fromfile(fname, "i2",
                                                       nSamples)]
                if DForm == 5:
                    # 32-bit samples read in 8 bits at a time
                    sid_rec["data"] = [np.fromfile(fname, "float32",
                                                   nSamples)]
                SID_rec = np.append(SID_rec, sid_rec, axis=0)
            else:
                SID_rec = np.delete(SID_rec, -1, 0)

    return(df_head, SID_spec, SID_rec)


def parse_SID(SID_spec, SID_records):
    """Parse AMX data in SID_records structure into matrices

    Parameters
    ----------
    SID_spec : numpy.ndarray
        Path to configuration file.
    SID_records : numpy.ndarray
        Path to configuration file.

    Returns
    -------
    tuple
        Tuple with (index, name in brackets):

        numpy.ndarray [0]
            audio array.
        numpy.ndarray [1]
            pressure/temperature array.
        numpy.ndarray [2]
            light (RGB) array.
        numpy.ndarray [3]
            IMU array.

    """
    audio = np.empty(0, dtype=np.int16)
    prtmp = np.empty(0, dtype=np.float32)
    rgb = np.empty(0, dtype=np.int16)
    imu = np.empty(0, dtype=np.int16)
    o2 = np.empty(0, dtype=np.int16)

    for rec in SID_records:
        sid_id = rec["SID_id"]
        sid = (SID_spec["SID"][sid_id]).upper()
        if sid.startswith("A"):
            audio = np.append(audio, rec["data"], axis=0)
        elif sid.startswith("P"):
            prtmp = np.append(prtmp, rec["data"], axis=0)
            prtmp_sid = sid_id
        elif sid.startswith("L"):
            rgb = np.append(rgb, rec["data"], axis=0)
            rgb_sid = sid_id
        elif sid.startswith("I") or sid.startswith("3"):
            imu = np.append(imu, rec["data"], axis=0)
            imu_sid = sid_id
        elif sid.startswith("O"):
            o2 = np.append(o2, rec["data"], axis=0)

    # Assemble IMU, RGB, and pressure/temperature matrices
    imu_accelx_cal = SID_spec["sensor"]["cal"][imu_sid][0]
    imu_accely_cal = SID_spec["sensor"]["cal"][imu_sid][1]
    imu_accelz_cal = SID_spec["sensor"]["cal"][imu_sid][2]
    imu_gyrox_cal = SID_spec["sensor"]["cal"][imu_sid][3]
    imu_gyroy_cal = SID_spec["sensor"]["cal"][imu_sid][4]
    imu_gyroz_cal = SID_spec["sensor"]["cal"][imu_sid][5]
    imu_magx_cal = SID_spec["sensor"]["cal"][imu_sid][6]
    imu_magy_cal = SID_spec["sensor"]["cal"][imu_sid][7]
    imu_magz_cal = SID_spec["sensor"]["cal"][imu_sid][8]

    imu_accelx = imu[0::9] * imu_accelx_cal
    imu_accely = imu[1::9] * imu_accely_cal
    imu_accelz = imu[2::9] * imu_accelz_cal
    imu_gyrox = imu[3::9] * imu_gyrox_cal
    imu_gyroy = imu[4::9] * imu_gyroy_cal
    imu_gyroz = imu[5::9] * imu_gyroz_cal
    imu_magx = imu[6::9] * imu_magx_cal
    imu_magy = imu[7::9] * imu_magy_cal
    imu_magz = imu[8::9] * imu_magz_cal

    imu_mat = np.column_stack((imu_accelx, imu_accely, imu_accelz,
                               imu_gyrox, imu_gyroy, imu_gyroz,
                               imu_magx, imu_magy, imu_magz))

    # RGB
    red_cal = SID_spec["sensor"]["cal"][rgb_sid][0]
    blue_cal = SID_spec["sensor"]["cal"][rgb_sid][1]
    green_cal = SID_spec["sensor"]["cal"][rgb_sid][2]

    red = rgb[0::3] * red_cal
    blue = rgb[1::3] * blue_cal
    green = rgb[2::3] * green_cal

    rgb_mat = np.column_stack((red, blue, green))

    # Pressure/temperature
    prsr_cal = SID_spec["sensor"]["cal"][prtmp_sid][0]
    tmp_cal = SID_spec["sensor"]["cal"][prtmp_sid][1]

    prsr = prtmp[0::2] * prsr_cal
    tmp = prtmp[1::2] * tmp_cal

    prtmp_mat = np.column_stack((prsr, tmp))

    return (audio, prtmp_mat, rgb_mat, imu_mat, o2)


if __name__ == '__main__':
    _DESCRIPTION = ("Parse binary AMX file from old Loggerhead "
                    "Instruments tags.")
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("amx_file", metavar="amx-file",
                        help="Path to *.amx file to parse")
    parser.add_argument("--ofigure-file",
                        default="motion_correct_compare.pdf",
                        type=argparse.FileType("w"),
                        help="Path to output figure file.")
    args = parser.parse_args()
    df_head, sid_spec, sid_rec = oAMX(args.amx_file)
    audio, prtmp, rgb, imu, _ = parse_SID(sid_spec, sid_rec)
