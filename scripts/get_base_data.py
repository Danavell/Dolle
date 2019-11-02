from pre_processing.utils import BaseDataFactory


BaseDataFactory.get_ladder_codes()
# for i in range(1, 6):
#     BaseDataFactory.factory(i, '01-01-18 to 01-01-19', fix_duplicates=True)

BaseDataFactory.factory(7, '01-01-18 to 01-01-19', fix_duplicates=True)
