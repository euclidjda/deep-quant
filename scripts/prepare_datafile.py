"Writes a .dat file according to user's specifications."

import argparse

from utils.data_utils import create_datafile


def main(datasource, ticlist, dest_basename):
    create_datafile(datasource, ticlist, dest_basename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "ticlist",
            help="""Specifies which tickers should be in the produced .dat
            file (NOTE: please postfix this with `.dat`)""")

    parser.add_argument(
            "dest_basename",
            help="""The name you would like the produced .dat file to have
            (NOTE: please postfix this with `.dat`)""")

    parser.add_argument(
            "--datasource",
            help="""Specifies where the data should come from (either
            `open_dataset` or `WRDS`""",
            default="open_dataset")

    args = parser.parse_args()
    main(args.datasource, args.ticlist, args.dest_basename)
