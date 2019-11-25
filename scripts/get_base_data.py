from pre_processing.utils import BaseDataFactory

BaseDataFactory.get_ladder_codes()
BaseDataFactory.factory(2, '28-02-16 to 2018-12-19', fix_duplicates=True, save=True)