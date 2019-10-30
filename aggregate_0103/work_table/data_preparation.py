from utils.work_table import data_preparation as dp
from utils.utils import get_dummy_products


class WorkTableCleaner:
    """
    removes duplicated rows, splits rows where breaks occurred and filters the work table
    to contain only the ladders in the ladder_filter
    """
    def __init__(self, work_table, stats, ladder_filter, remove_overlaps):
        self.work_table = work_table
        self.stats = stats
        self._ladder_filter = ladder_filter
        self._remove_overlaps = remove_overlaps

    def filter(self):
        self.work_table = dp.add_columns(self.work_table)
        _, self.work_table = get_dummy_products(self.work_table)
        self.work_table = self._ladder_filter(self.work_table)

    def clean(self):
        self.work_table = self._remove_overlaps(self.work_table, self.stats)

    def remove_breaks(self, sensor_data, breaks):
        return dp.remove_breaks(breaks, self.work_table, sensor_data, self.stats)

