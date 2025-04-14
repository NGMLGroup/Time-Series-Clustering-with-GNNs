import os
from datetime import datetime, timedelta
from functools import reduce
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tsl
from tqdm import tqdm

from tsl.datasets import DatetimeDataset
from tsl.utils import download_url, extract_zip

START = datetime(2008, 12, 31, 0, 0)
ID_COL = 'id'
TARGET_COL = 'load'
DATETIME_COL = 'datetime'
SAMPLES_PER_DAY = 48


def parse_date(date):
    """
    Parses date strings for the irish dataset.

    :param date: timestamp (see dataset description for information)
    :return: datetime
    """
    return START + timedelta(days=date // 100) + timedelta(
        hours=0.5 * (date % 100))


class FilteredCER(DatetimeDataset):

    url = None # request url at https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

    default_freq = '30T'

    def __init__(self,
                 root=None,
                 freq=None,
                 print_dataset_info=True,
                 missing_threshold=0.05,
                 time_cutoff=None,
                 remove_other=True,
                 resample_hourly=True):
        # set root path
        if root is None:
            self.root = None # TODO: set default path
        else:
            self.root = root

        self.mthrsh = missing_threshold
        self.tcutoff = time_cutoff

        self.print_dataset_info = print_dataset_info
        self.remove_other = remove_other
        self.resample_hourly = resample_hourly

        # load dataset
        df, mask = self.load()
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='correntropy',
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         name='CER-FILT')

    @property
    def raw_file_names(self):
        return [f'File{i}.txt.zip' for i in range(1, 7)] + \
               ['allocations.xlsx', 'manifest.docx']

    @property
    def required_file_names(self):
        return ['cer_en.h5', 'allocations.xlsx', 'manifest.docx']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)
        downloaded_folder = os.path.join(self.root_dir, 'irish')
        # move files to root folder
        for file in os.listdir(downloaded_folder):
            if file in self.raw_file_names:
                os.rename(os.path.join(downloaded_folder, file),
                          os.path.join(self.root_dir, file))
        self.clean_root_dir()

    def build(self):
        self.maybe_download()
        # Build dataset
        tsl.logger.info("Building the dataset...")
        dfs = []
        # read csv from zip files
        for filepath in tqdm(filter(lambda x: '.zip' in x,
                                    os.listdir(self.root_dir))):
            filepath = os.path.join(self.root_dir, filepath)
            zip = ZipFile(filepath)
            ifile = zip.open(zip.infolist()[0])
            data = pd.read_csv(ifile, sep=" ", header=None,
                               names=[ID_COL, DATETIME_COL, TARGET_COL])
            data = data.apply(pd.to_numeric)
            data = pd.pivot_table(data, values=TARGET_COL, index=[DATETIME_COL],
                                  columns=[ID_COL])
            dfs.append(data)
        # merge dfs
        df = reduce(lambda left, right: pd.merge(left, right, on=DATETIME_COL),
                    dfs)

        # parse datetime index
        df = df.reset_index()
        ts = df[DATETIME_COL].values % 100
        # remove inconsistent timestamps
        df = df[(ts > 0) & (ts <= SAMPLES_PER_DAY)]
        index = pd.to_datetime(df[DATETIME_COL].apply(parse_date))
        df.loc[:, DATETIME_COL] = index
        df = df.drop_duplicates(DATETIME_COL)
        df = df.set_index(DATETIME_COL).astype('float32')

        # save df
        path = os.path.join(self.root_dir, 'cer_en.h5')
        df.to_hdf(path, key='data', complevel=3)
        self.clean_downloads()

    def load(self):
        # Load raw data
        df = self.load_raw()
        tsl.logger.info('Loaded raw dataset.')
        # Fix missing timestamps
        df = df.asfreq(self.default_freq)
        mask = ~np.isnan(df.values)
        df = df.fillna(0.)

        if self.print_dataset_info:
            org_allocs = pd.read_excel(os.path.join(self.root_dir, 'allocations.xlsx'))
            org_codes = np.array(org_allocs['Code'].to_list())
            print(f'[Original] residential: {(org_codes==1).sum()}, sme: {(org_codes == 2).sum()}, other: {(org_codes == 3).sum()}')
            print(f'[Original] Number of time steps: {df.shape[0]}')

        # Resample to hourly data
        if self.resample_hourly:
            df = df.resample('H').mean()
            temp_mask = np.zeros((df.shape[0], df.shape[1]), dtype=bool)
            temp_mask[0, :] = mask[0, :]
            temp_mask[1:, :] = np.logical_and(mask[::2, :], mask[1::2, :])
            mask = temp_mask
            if self.print_dataset_info:
                print(f'Resampled to hourly data: {df.shape[0]} time steps')

        # Drop last time steps (optional)
        if self.tcutoff is not None:
            mask = mask[:self.tcutoff,:]
            df = df.iloc[:self.tcutoff,:]
        else:
            self.tcutoff = mask.shape[0]

        # Remove nodes with too many missing values
        idx_keep = mask.sum(axis=0) > int((1.0-self.mthrsh)*self.tcutoff)
        if self.print_dataset_info:
            print(f"Dropping {(idx_keep==False).sum()} nodes ({(idx_keep==False).sum()/mask.shape[1]*100:.2f}%)")
        mask = mask[:,idx_keep]
        df = df.loc[:,idx_keep]

        # Get codes
        id_ds = list(df.columns)
        allocs = pd.read_excel(os.path.join(self.root_dir, 'allocations.xlsx'))
        id_alloc = [np.where(allocs['ID'] == i)[0][0] for i in id_ds]
        codes = np.array(allocs['Code'].to_list())[id_alloc]
        id_ds = np.array(id_ds)

        # Divide data by codes
        id_residential = id_ds[codes==1]
        id_sme = id_ds[codes==2]
        id_other = id_ds[codes==3]

        # Filter out other nodes (optional)
        id_other_filt = np.array([]) if self.remove_other else id_other
        if self.print_dataset_info:
            print(f'Removed {id_other.shape[0]} other nodes')

        if self.print_dataset_info:
            print(f'[Filtered] residential: {id_residential.shape[0]} ({id_residential.shape[0]/id_residential.shape[0]*100:.1f} %), '
                f'sme: {id_sme.shape[0]} ({id_sme.shape[0]/id_sme.shape[0]*100:.1f} %), '
                f'other: {id_other_filt.shape[0]} ({id_other_filt.shape[0]/id_other.shape[0]*100:.1f} %)')
            print(f'[Filtered] Number of time steps: {df.shape[0]}')

        # Reindex stuff
        id_filt = list(np.hstack((id_residential, id_sme, id_other_filt)))
        col_idx = [df.columns.get_loc(i) for i in id_filt]
        df = df.reindex(columns=id_filt)
        mask = mask[:, col_idx] if mask is not None else None
        self.codes = codes[col_idx]

        return df, mask

    def load_raw(self) -> pd.DataFrame:
        self.maybe_build()
        return pd.read_hdf(self.required_files_paths[0])

if __name__ == '__main__':
    root = os.path.join(os.getcwd(), '..', '..', 'datasets', 'cer')
    cer = FilteredCER(root=root)
    print(cer)