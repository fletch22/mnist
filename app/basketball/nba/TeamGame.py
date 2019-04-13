import logging
from app.basketball.nba.Player import Player
from app.basketball.nba.TeamGameStatTotals import TeamGameStatTotals
from app.util.object_utils import UNSET, fix_illegal_keys


class TeamGame:
    score = UNSET
    name = UNSET
    totals = UNSET
    players = UNSET

    def __init__(self, dict_loader):
        fixed_dict = fix_illegal_keys(dict_loader)
        self.__dict__ = fixed_dict

        self.totals = TeamGameStatTotals(fixed_dict['totals'])
        self.score = fixed_dict['scores']['T']
        del self.scores
        self.players = load_players(dict_loader, "players")


def load_players(dict_thingy, key):
    players = []
    dict_players = dict_thingy[key]
    for player_key in dict_players.keys():
        player = Player(dict_players[player_key])
        player.name = player_key
        players.append(player)

    missing_players = 12 - len(players)

    for i in range(missing_players):
        players.append(Player(None))

    return players
