import time
from datetime import timedelta

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Literal, Union
from collections import deque, OrderedDict
from queue import PriorityQueue


@dataclass(order=True)
class Order:  # Our own placed order
    timestamp: int
    order_id: int
    side: str
    size: float
    price: float
    

@dataclass(order=True)
class CancelOrder:
    timestamp: int
    order_id: int


@dataclass(order=True)
class AnonTrade:  # Market trade
    timestamp: int
    side: str
    size: float
    price: str
    

@dataclass
class OwnTrade:  # Execution of own placed order
    timestamp: int
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __eq__(self, other):
        return self.timestamp == other.timestamp
    

@dataclass(order=True)
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    timestamp: int
    asks: list[tuple[float, float]]  # tuple[price, size]
    bids: list[tuple[float, float]]
    
    
@dataclass
class MdUpdate:  # Data of a tick
    timestamp: int
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None


    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __eq__(self, other):
        return self.timestamp == other.timestamp


def load_md_from_files(lobs_path: str, trades_path: str, n_orderbook_levels: int = 10) -> list[MdUpdate]:

    trades = pd.read_csv(trades_path).sort_values(by='exchange_ts').iloc[:100000]
    lobs = pd.read_csv(lobs_path).sort_values(by='exchange_ts').iloc[:150000]
    tc = {col: i for i, col in enumerate(trades.columns)}
    lc = {col: i for i, col in enumerate(lobs.columns)}
    trades = trades.to_numpy()
    lobs = lobs.to_numpy()
    
    md_updates = []
    trades_idx, lobs_idx = 0, 0
    while trades_idx < len(trades) and lobs_idx < len(lobs):
        orderbook, trades_ = None, None
        exchange_ts = None

        if trades_idx == len(trades) or lobs[lobs_idx][lc['exchange_ts']] <= trades[trades_idx][tc['exchange_ts']]:
            row = lobs[lobs_idx]
            orderbook = OrderbookSnapshotUpdate(
                timestamp=lobs[lobs_idx][lc['receive_ts']],
                bids=[(row[lc[f'bid_price_{i}']], row[lc[f'bid_vol_{i}']]) for i in range(n_orderbook_levels)],
                asks=[(row[lc[f'ask_price_{i}']], row[lc[f'ask_vol_{i}']]) for i in range(n_orderbook_levels)],
            )
            exchange_ts = row[lc['exchange_ts']]
            lobs_idx += 1
        
        if lobs_idx == len(lobs) or trades[trades_idx][tc['exchange_ts']] <= lobs[lobs_idx][lc['exchange_ts']]:
            trades_ = []
            while True:
                trades_.append(AnonTrade(
                    timestamp=trades[trades_idx][tc['receive_ts']],
                    size=trades[trades_idx][tc['size']],
                    price=trades[trades_idx][tc['price']],
                    side=trades[trades_idx][tc['aggro_side']]
                ))

                is_next_valid = trades_idx + 1 < len(trades) and \
                    trades[trades_idx + 1][tc['exchange_ts']] == trades[trades_idx][tc['exchange_ts']]
                if not is_next_valid:
                    break 
                trades_idx += 1

            exchange_ts = trades[trades_idx][tc['exchange_ts']]
            trades_idx += 1
        
        md_updates.append(MdUpdate(timestamp=exchange_ts, orderbook=orderbook, trades=trades_))

    return md_updates


class Sim:
    def __init__(self, execution_latency: float, md_latency: float) -> None:
        self.md_queue = deque(load_md_from_files(
                "md/btcusdt:Binance:LinearPerpetual/lobs.csv", 
                "md/btcusdt:Binance:LinearPerpetual/trades.csv"
                ))
        
        self.actions_queue = deque()
        self.strategy_updates_queue = PriorityQueue()
        self.active_orders = {}
        self.orders = {}
        self.best_bid = 0
        self.best_ask = 1e18
        self.receive_ts = 0
        self.exchange_ts = 0
        self.stop = False

        md_update = None
        while True:
            md_update = self.md_queue.popleft()
            if md_update.orderbook is not None: 
                break
        
        self.tick_md(md_update)
        self.order_cnt = 0
        self.trade_cnt = 0
        self.execution_latency = execution_latency * 10**6
        self.md_latency = md_latency * 10**6


    def tick(self) -> Union[MdUpdate, OwnTrade]:
        
        if len(self.md_queue) == 0:
            self.stop = True
            if not self.strategy_updates_queue.empty():
                return self.return_update()
            
            raise StopIteration
            

        while True:
            strategy_time = 1e27 if self.strategy_updates_queue.empty() else self.strategy_updates_queue.queue[0].timestamp
            md_time = 1e28 if len(self.md_queue) == 0 else self.md_queue[0].timestamp
            actions_time = 1e28 if len(self.actions_queue) == 0 else self.actions_queue[0].timestamp
            if strategy_time < md_time and strategy_time < actions_time:
                return self.return_update()
            
            if md_time < actions_time:
                self.tick_md(self.md_queue.popleft())
            else:
                self.prepare_order(self.actions_queue.popleft())

            self.execute_orders()        


    def return_update(self) -> Union[MdUpdate, OwnTrade]:
        if self.strategy_updates_queue.empty():
            return None
        else:
            update = self.strategy_updates_queue.get()
            self.receive_ts = update.timestamp
        return update

    
    def execute_orders(self):
        executed = []
        for order in self.active_orders.values():
            if order.side == 'BID' and order.price >= self.best_ask or order.side == 'ASK' and order.price <= self.best_bid:
                executed.append(order.order_id)
                own_trade = OwnTrade(
                    timestamp=self.exchange_ts+self.md_latency, 
                    trade_id=self.trade_cnt,
                    order_id=order.order_id,
                    price=self.best_ask if order.side == 'BID' else self.best_bid, 
                    size=order.size,
                    side=order.side,
                    )

                self.trade_cnt += 1
                self.strategy_updates_queue.put(own_trade)
        
        for order_id in executed:
            self.active_orders.pop(order_id)


    def place_order(self, price: float, size: float, side: Literal['BID', 'ASK']) -> Optional[Order]:
        if self.stop:
            return None
        
        order = Order(
            timestamp=self.receive_ts+self.execution_latency, 
            order_id=self.order_cnt, 
            price=price,
            size=size,
            side=side
        )
        self.order_cnt += 1
        self.actions_queue.append(order)
        return order


    def cancel_order(self, order_id: int):
        self.actions_queue.append(
            CancelOrder(timestamp=self.receive_ts + self.execution_latency, order_id=order_id)
        )


    def tick_md(self, md_update):
        ts = 0
        if md_update.trades is not None:
            for trade in md_update.trades:
                if trade.side == 'BID' and trade.price > self.best_bid:
                    self.best_bid = trade.price
                    ts = max(ts, trade.timestamp)
                elif trade.side == 'ASK' and trade.price < self.best_ask:
                    self.best_ask = trade.price
                    ts = max(ts, trade.timestamp)

        if md_update.orderbook is not None: 
            self.best_bid = max(self.best_bid, md_update.orderbook.bids[0][0])
            self.best_ask = max(self.best_ask, md_update.orderbook.asks[0][0])
            ts = max(ts, md_update.orderbook.timestamp)

        self.exchange_ts = md_update.timestamp
        md_update.timestamp = ts
        self.strategy_updates_queue.put(md_update)


    def prepare_order(self, action: Union[Order, int]):
        if isinstance(action, CancelOrder):
            self.active_orders.pop(action.order_id, 0)
        else:
            self.active_orders[action.order_id] = action



@dataclass
class Strategy:
    max_postion: float
    maker_fee: float
    t_0: int


    def run(self, sim: Sim):
        self.best_bid = 0
        self.best_ask = 1e18
        size = 0.01
        position = 0.
        placed_orders = {}
        balance = 0.
        while True:
            try:
                update = sim.tick()

                if update is None:
                    break 

                cur_timestamp = update.timestamp

                if isinstance(update, MdUpdate):
                    self.tick_md(update)
                else:
                    if update.side == 'BID':
                        position += update.size
                        balance -= update.price * update.size
                    else:
                        position -= update.size
                        balance += update.price * update.size

                    balance -= self.maker_fee * update.price * update.size
                    total_size += update.size

                    placed_orders.pop(update.order_id, 0)
                    continue
                
                # canceling orders
                canceled = []
                for order_id, order in placed_orders.items():
                    if cur_timestamp - order.timestamp > self.t_0:
                        sim.cancel_order(order_id)
                        canceled.append(order_id)

                for order_id in canceled:
                    placed_orders.pop(order_id)

                # placing orders
                side = np.random.choice(['BID', 'ASK'])

                order = None
                if side == 'BID':
                    if position + size > self.max_postion:
                        side = 'ASK'
                    else:
                        order = sim.place_order(price=self.best_bid, size=size, side=side)

                if side == 'ASK':
                    order = sim.place_order(price=self.best_ask, size=size, side=side)
                
                if order is not None:
                    placed_orders[order.order_id] = order

            except StopIteration:
                break
            
        return balance + (self.best_ask if position < 0 else self.best_bid) * position
    

    def tick_md(self, md_update):
        if md_update.trades is not None:
            for trade in md_update.trades:
                if trade.side == 'BID':
                    self.best_bid = max(self.best_bid, trade.price)
                else:
                    self.best_ask = min(self.best_ask, trade.price)

        if md_update.orderbook is not None: 
            self.best_bid = max(self.best_bid, md_update.orderbook.bids[0][0])
            self.best_ask = min(self.best_ask, md_update.orderbook.asks[0][0])
    

    

if __name__ == "__main__":
    sim = Sim(10, 10)
    strategy = Strategy(max_postion=1., maker_fee=0., t_0=1000)
    print(strategy.run(sim))
