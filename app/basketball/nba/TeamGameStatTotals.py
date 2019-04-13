from app.util.object_utils import UNSET, fix_illegal_keys


class TeamGameStatTotals:
    FT = UNSET
    twoPA = UNSET
    FG = UNSET
    DRB = UNSET
    ORB_pct = UNSET
    AST = UNSET
    threePAr = UNSET
    PF = UNSET
    FGA = UNSET
    DRBr = UNSET
    twoP = UNSET
    ORBr = UNSET
    TOV_pct = UNSET
    AST_pct = UNSET
    FTAr = UNSET
    FIC = UNSET
    eFG_pct = UNSET
    FG_pct = UNSET
    twoPAr = UNSET
    plus_minus = UNSET
    USG_pct = UNSET
    DRtg = UNSET
    twoP_pct = UNSET
    DRB_pct = UNSET
    ORtg = UNSET
    TRB_pct = UNSET
    ORB = UNSET
    threeP = UNSET
    TOV = UNSET
    STL_TOV = UNSET
    TSA = UNSET
    AST_TOV = UNSET
    threePA = UNSET
    BLK_pct = UNSET
    FT_pct = UNSET
    PTS = UNSET
    HOB = UNSET
    STL = UNSET
    TRB = UNSET
    FTA = UNSET
    BLK = UNSET
    FTr = UNSET
    TS_pct = UNSET
    FT_FGA = UNSET
    threeP_pct = UNSET
    STL_pct = UNSET

    def __init__(self, dict_loader):
        fixed_dict = fix_illegal_keys(dict_loader)
        self.__dict__ = fixed_dict
