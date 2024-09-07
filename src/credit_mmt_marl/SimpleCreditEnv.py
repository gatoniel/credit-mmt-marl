import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Tuple, Box, Dict

from pettingzoo import ParallelEnv


# OBSERVATION dict items
PARAMS = "parameter"
RESOURCES = "resources"
ACCOUNTS = "accounts"
LAST_PRICES = "last prices"

# ACTION dict items
USE_GOODS = "use goods"
USE_CAPITAL = "use capital"
BUY_GOODS = "buy goods"
BUY_CAPITAL = "buy capital"
SELL_GOODS = "sell goods"
SELL_CAPITAL = "sell capital"
SELL_TRADED_CAPITAL = "sell traded capital"

BUY_GOODS_PRICE = "buy goods price"
BUY_CAPITAL_PRICE = "buy capital price"
SELL_GOODS_PRICE = "sell goods price"
SELL_CAPITAL_PRICE = "sell capital price"


class SimpleCreditEnvV0(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "Simple Credit Environment v0",
    }

    def __init__(self, num_std_entities: int = 2, seed=20240906):
        """The init method takes in environment arguments."""
        # Financial variables
        self.accounts = None
        self.liabilities = None  # exclude for the moment
        # Real resource variables
        self.goods = None
        self.capital = None
        self.traded_goods = None
        self.traded_capital = None
        # Production parameters
        self.alpha = None
        self.beta = None
        # Market operations
        self.demand_prices_goods = None
        self.offer_prices_goods = None
        self.demand_prices_capital = None
        self.offer_prices_capital = None

        self.timestep = None
        self.num_std_entities = num_std_entities
        self.possible_agents = [f"Player {i}" for i in range(num_std_entities)]

        # Random Generator
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """

        self.agents = copy(self.possible_agents)
        self.num_players = len(self.agents)
        num_players = self.num_players
        self.timestep = 0

        # Financial variables
        self.accounts = np.zeros((num_players, num_players))
        # Real resource variables
        self.goods = np.zeros(num_players)
        self.capital = np.ones(num_players)
        self.traded_goods = np.zeros(num_players)
        self.traded_capital = np.zeros(num_players)

        # Production parameters
        self.alpha = self.rng.uniform(0.1, 0.9, num_players)
        self.beta = self.rng.uniform(0.1, 0.9, num_players)

        # Market operations
        self.demand_prices_goods = np.zeros((num_players, num_players))
        self.offer_prices_goods = np.zeros((num_players, num_players))
        self.demand_prices_capital = np.zeros((num_players, num_players))
        self.offer_prices_capital = np.zeros((num_players, num_players))

        observations = self.get_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def get_observations(self):
        # resources are length 4 for each player
        resources = np.stack(
            [self.goods, self.capital, self.traded_goods, self.traded_capital],
            axis=-1,
        )
        # parameters are length 2 for each player
        parameters = np.stack([self.alpha, self.beta], axis=-1)
        # potentially do not hide information here via sorting?
        # prices has length 4 * num_players**2
        prices = np.concatenate(
            [
                np.sort(x, axis=None)
                for x in [
                    self.demand_prices_goods,
                    self.demand_prices_capital,
                    self.offer_prices_goods,
                    self.offer_prices_capital,
                ]
            ]
        )

        return {
            self.agents[i]: {
                PARAMS: parameters[i],
                RESOURCES: resources[i],
                ACCOUNTS: self.accounts[i],
                LAST_PRICES: prices,
            }
            for i in range(self.num_players)
        }

    def using_resources(self, actions, resource):
        clip = {
            USE_GOODS: self.traded_goods,
            USE_CAPITAL: self.traded_capital,
            BUY_GOODS: np.inf,
            BUY_CAPITAL: np.inf,
            SELL_GOODS: self.goods + self.traded_goods,
            SELL_CAPITAL: self.capital,
            SELL_TRADED_CAPITAL: self.traded_capital,
        }[resource]
        using_ = np.array(
            [actions[f"Player {i}"][resource] for i in range(self.num_players)]
        ).squeeze()
        return np.clip(using_, 0, clip)

    def get_price_list(self, actions, resource):
        return np.stack(
            [actions[f"Player {i}"][resource] for i in range(self.num_players)],
            axis=0,
        ).squeeze()

    def get_order_list(self, demand, order):
        rel_prices = np.expand_dims(demand, axis=1) / np.expand_dims(
            order, axis=0
        )
        num_greater_1 = np.sum(rel_prices >= 1)
        inds = np.argsort(rel_prices, axis=None)[::-1]
        un_inds = np.stack(
            np.unravel_index(inds[:num_greater_1], (self.num_players,) * 3),
            axis=1,
        )
        diff_seller_buyer = un_inds[:, 0] != un_inds[:, 1]
        return un_inds[diff_seller_buyer, :]

    def get_exchange(self, buyer, seller, currency, prices, demand, offer):
        unit_price = prices[buyer, currency]
        # What is the maximal amount the buyer is able to buy
        if buyer == currency:
            demand_possible = np.inf
        else:
            buyer_account = self.accounts[buyer, currency]
            demand_possible = buyer_account / unit_price
        # How much does he WANT to buy?
        # max(min(...)) makes sure that we don't get negative demands
        demand = max(min(demand_possible, demand[buyer]), 0)
        # How much can be offered by seller?
        exchange = max(min(demand, offer[seller]), 0)
        final_price = exchange * unit_price
        return final_price, exchange

    def resource_transaction(
        self, buyer, seller, exchange, offer, sold, traded, demand
    ):
        offer[seller] -= exchange  # how much remains to sell
        sold[seller] += exchange  # how much needs to be substracted later
        traded[buyer] += exchange
        demand[buyer] -= exchange  # how much remains to be bought

    def total_resources(self):
        return (
            self.goods.sum()
            + self.traded_goods.sum()
            + self.capital.sum()
            + self.traded_capital.sum()
        )

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """

        num_players = self.num_players

        # Update new production of goods
        energy = self.rng.exponential(scale=1.0, size=num_players)
        produced_goods = self.capital**self.alpha * energy ** (1 - self.alpha)
        if np.any(np.isnan(produced_goods)):
            print(self.capital, energy)

        self.goods += produced_goods

        # Update new production of capital
        using_goods = self.using_resources(actions, USE_GOODS)
        using_capital = self.using_resources(actions, USE_CAPITAL)

        new_capital = using_goods**self.beta * using_capital ** (
            1 - self.beta
        )
        if np.any(np.isnan(new_capital)):
            print(using_goods, using_capital)

        self.capital += new_capital
        self.traded_goods = np.clip(
            self.traded_goods - using_goods, 0.0, np.inf
        )
        self.traded_capital = np.clip(
            self.traded_capital - using_capital, 0.0, np.inf
        )

        # Trading
        available_goods = self.using_resources(actions, SELL_GOODS)
        available_capital = self.using_resources(actions, SELL_CAPITAL)
        available_traded_capital = self.using_resources(
            actions, SELL_TRADED_CAPITAL
        )

        self.demand_prices_goods = self.get_price_list(actions, BUY_GOODS_PRICE)
        self.demand_prices_capital = self.get_price_list(
            actions, BUY_CAPITAL_PRICE
        )
        self.offer_prices_goods = self.get_price_list(actions, SELL_GOODS_PRICE)
        self.offer_prices_capital = self.get_price_list(
            actions, SELL_CAPITAL_PRICE
        )

        demand_goods = self.using_resources(actions, BUY_GOODS)
        sold_goods = np.zeros(num_players)
        revenue = np.zeros((num_players, num_players))

        goods_order_list = self.get_order_list(
            self.demand_prices_goods, self.offer_prices_goods
        )
        capital_order_list = self.get_order_list(
            self.demand_prices_capital, self.offer_prices_capital
        )
        for buyer, seller, currency in goods_order_list:
            final_price, exchange = self.get_exchange(
                buyer,
                seller,
                currency,
                self.demand_prices_goods,
                demand_goods,
                available_goods,
            )

            # Financial transactions
            self.accounts[buyer, currency] -= final_price
            revenue[seller, currency] += final_price

            # Resource transaction
            available_goods[seller] -= exchange  # how much remains to sell
            sold_goods[
                seller
            ] += exchange  # how much needs to be substracted later
            self.traded_goods[buyer] += exchange
            demand_goods[buyer] -= exchange  # how much remains to buy
            self.resource_transaction(
                buyer,
                seller,
                exchange,
                available_goods,
                sold_goods,
                self.traded_goods,
                demand_goods,
            )

        demand_capital = self.using_resources(actions, SELL_GOODS)
        sold_capital = np.zeros(num_players)
        total_available_capital = available_capital + available_traded_capital
        for buyer, seller, currency in capital_order_list:
            final_price, exchange = self.get_exchange(
                buyer,
                seller,
                currency,
                self.demand_prices_capital,
                demand_capital,
                total_available_capital,
            )

            # Financial transactions
            self.accounts[buyer, currency] -= final_price
            revenue[seller, currency] += final_price

            # Resource transaction
            self.resource_transaction(
                buyer,
                seller,
                exchange,
                total_available_capital,
                sold_capital,
                self.traded_capital,
                demand_capital,
            )

        # Only now add revenue. Otherwise early sellers could buy with more
        # money later in the process. Similarly, sellers of goods could be
        # buying more capital.
        self.accounts += revenue
        # First substract sold goods from goods
        goods_sub = self.goods - sold_goods
        # Now substract oversold goods from traded goods
        inds = np.argwhere(goods_sub < 0)
        self.traded_goods[inds] -= goods_sub[inds]
        # Set negative goods to zero
        self.goods = np.clip(goods_sub, 0, np.inf)

        # First sell capital from oversupplied capital stock. Then, sell it
        # from both internal stocks to same degree.
        diff = np.clip(
            np.abs(available_capital - available_traded_capital),
            0,
            sold_capital,
        )
        halving = np.clip(sold_capital - diff, 0, np.inf) / 2
        inds1 = available_capital > available_traded_capital
        self.capital[inds1] -= diff[inds1]
        inds2 = np.logical_not(inds1)
        self.traded_capital[inds2] -= diff[inds2]
        self.capital = np.clip(self.capital - halving, 0, np.inf)
        self.traded_capital = np.clip(self.traded_capital - halving, 0, np.inf)

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {f"Player {i}": self.capital[i] for i in range(num_players)}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 200:
            truncations = {a: True for a in self.agents}
        self.timestep += 1

        # Get observations
        observations = self.get_observations()
        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        for i in range(self.num_players):
            print(self.goods)
            print(self.traded_goods)
            print(self.capital)
            print(self.traded_capital)
            print(self.accounts)
            print("\n")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(
            {
                PARAMS: Box(0.0, 1.0, shape=(2,)),
                RESOURCES: Box(0.0, np.inf, shape=(4,)),
                ACCOUNTS: Box(-np.inf, np.inf, shape=(self.num_players,)),
                LAST_PRICES: Box(
                    0.0, np.inf, shape=(4 * self.num_players**2,)
                ),
            }
        )

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict(
            {
                USE_GOODS: Box(0.0, np.inf),
                USE_CAPITAL: Box(0.0, np.inf),
                BUY_GOODS: Box(0.0, np.inf),
                BUY_CAPITAL: Box(0.0, np.inf),
                SELL_GOODS: Box(0.0, np.inf),
                SELL_CAPITAL: Box(0.0, np.inf),
                SELL_TRADED_CAPITAL: Box(0.0, np.inf),
                BUY_GOODS_PRICE: Box(0.0, np.inf, shape=(self.num_players,)),
                SELL_GOODS_PRICE: Box(0.0, np.inf, shape=(self.num_players,)),
                BUY_CAPITAL_PRICE: Box(0.0, np.inf, shape=(self.num_players,)),
                SELL_CAPITAL_PRICE: Box(0.0, np.inf, shape=(self.num_players,)),
            }
        )
