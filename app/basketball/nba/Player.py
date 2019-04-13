import time
from calendar import timegm

from app.util.object_utils import fix_illegal_keys


def get_unix_epoch_time(the_date):
    utc_time = time.strptime(the_date, "%Y-%m-%d")
    return timegm(utc_time)


UNSET_BIRTHDAY = get_unix_epoch_time("1970-01-01")


class Player:
    STL_pct = 0.0
    blank = ""
    FT = 0
    weight = 0
    threeP = 0
    TOV = 0
    STL_TOV = 0
    TSA = 0
    twoPA = 0
    college = ""
    FG = 0
    threePA = 0
    DRB = 0
    ORB_pct = 0
    BLK_pct = 0
    AST_TOV = 0
    position = ""
    AST = 0
    FT_pct = 0
    threePAr = 0
    PF = 0
    PTS = 0
    FGA = 0
    DRBr = 0
    ORBr = 0
    twoP = 0
    STL = 0
    TRB = 0
    TOV_pct = 0
    AST_pct = 0
    FTAr = 0
    FTA = 0
    FIC = 0
    eFG_pct = 0
    BLK = 0
    birth_date = UNSET_BIRTHDAY
    FG_pct = 0
    twoPAr = 0
    FTr = 0
    plus_minus = 0
    name = ""
    USG_pct = 0
    DRB_pct = 0
    TS_pct = 0
    experience = 0
    height = 0
    twoP_pct = 0
    MP = 0
    DRtg = 0
    ORtg = 0
    TRB_pct = 0
    FT_FGA = 0
    ORB = 0
    threeP_pct = 0
    HOB = 0

    def __init__(self, dict_loader=None):

        if dict_loader is not None:
            fixed_dict = fix_illegal_keys(dict_loader)
            self.__dict__ = fixed_dict

            self.birth_date = get_unix_epoch_time(self.birth_date)

            self.weight = float(self.weight)

        if self.MP is None:
            self.MP = 0.0
