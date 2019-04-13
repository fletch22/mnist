from calendar import timegm
import time

from app.basketball.nba.TeamGame import TeamGame
from app.util.object_utils import UNSET, fix_illegal_keys


class Game:
    code = UNSET
    season = UNSET
    away = UNSET
    time = UNSET
    country = UNSET
    date = UNSET
    home = UNSET
    type = UNSET

    def __init__(self, dict_loader):
        fixed_dict = fix_illegal_keys(dict_loader)
        self.__dict__ = fixed_dict

        utc_time = time.strptime(self.date, '%Y-%m-%d')
        self.date = timegm(utc_time)

        self.home = TeamGame(self.home)
        self.away = TeamGame(self.away)
