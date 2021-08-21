import pandas as pd
import numpy as np


class KaggleTitanic:

    def main(self):
        df = self.data_read()

    def data_read(self):
        return pd.read_csv('data/train.csv')


if __name__ == '__main__':
    kaggle_titanic = KaggleTitanic()
    kaggle_titanic.main()
