"""
xtquant Linux 兼容层

提供在无法安装官方 xtquant 库的环境下的最小兼容实现，
优先尝试调用 akshare 获取行情数据，并在 akshare 不可用时
返回空数据以保证程序能够继续运行。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_HAS_XTQUANT = False

try:  # pragma: no cover - 仅在官方库可用时运行
    from xtquant import xtdata as _xtdata  # type: ignore
    from xtquant.xttrader import (  # type: ignore
        XtQuantTrader as _XtQuantTrader,
        XtQuantTraderCallback as _XtQuantTraderCallback,
    )
    from xtquant import xtconstant as _xtconstant  # type: ignore
    from xtquant.xttype import StockAccount as _StockAccount  # type: ignore

    _HAS_XTQUANT = True
except Exception:  # pragma: no cover - 用于兼容缺失 xtquant 的环境
    _HAS_XTQUANT = False


def has_xtquant() -> bool:
    """判断当前环境是否安装了官方 xtquant 库。"""
    return _HAS_XTQUANT


if _HAS_XTQUANT:
    # 若官方库可用，直接透传
    xtdata = _xtdata  # type: ignore
    XtQuantTrader = _XtQuantTrader  # type: ignore
    XtQuantTraderCallback = _XtQuantTraderCallback  # type: ignore
    xtconstant = _xtconstant  # type: ignore
    StockAccount = _StockAccount  # type: ignore
else:
    try:  # pragma: no cover - 仅在 akshare 可用时运行
        import akshare as ak
    except Exception as exc:  # pragma: no cover - akshare 不可用
        ak = None  # type: ignore
        logger.warning("akshare 未安装或不可用，将仅提供空数据兼容实现: %s", exc)

    @dataclass
    class _StockAccountStub:
        account_id: str
        account_type: str

    class _XtConstantStub:
        STOCK_BUY = 1
        STOCK_SELL = 2
        SECURITY_ACCOUNT = "SECURITY_ACCOUNT"
        FIX_PRICE = "FIX_PRICE"
        ORDER_SUCCEEDED = "ORDER_SUCCEEDED"
        DIRECTION_FLAG_LONG = "DIRECTION_FLAG_LONG"
        OFFSET_FLAG_OPEN = "OFFSET_FLAG_OPEN"
        OFFSET_FLAG_CLOSE = "OFFSET_FLAG_CLOSE"

    class XtQuantTraderCallback:  # pragma: no cover - 仅供继承使用
        """交易回调基类占位，实现空方法以保证继承兼容。"""

        def __init__(self) -> None:
            pass

    class XtQuantTrader:  # pragma: no cover - 实盘功能不可用时给出提醒
        """交易接口占位实现，提示用户当前环境不可用。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "XtQuantTrader 在 Linux 兼容模式下不可用，请在具备 xtquant 环境的系统中使用。"
            )

    class _XtDataCompat:
        """xtdata 兼容实现，优先使用 akshare 获取行情。"""

        _MINUTE_PERIOD_MAP = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "60m": "60",
        }

        _SECTOR_FILE_MAP = {
            "上证A股": "上证A股_股票列表.csv",
            "深证A股": "深证A股_股票列表.csv",
            "创业板": "创业板_股票列表.csv",
            "科创板": "科创板_股票列表.csv",
            "沪深A股": "沪深A股_股票列表.csv",
            "指数": "指数_股票列表.csv",
            "全部股票": "全部股票_股票列表.csv",
            "沪深300": "沪深300成分股_股票列表.csv",
            "中证500": "中证500成分股_股票列表.csv",
            "上证50": "上证50成分股_股票列表.csv",
        }

        def __init__(self) -> None:
            self.logger = logging.getLogger("kh_xtquant_compat.xtdata")
            self.data_dir = os.path.join(os.path.dirname(__file__), "data")

        # --------- 对外公开方法 -----------
        def download_history_data(
            self,
            stock_code: str,
            period: str = "1d",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            **_: Any,
        ) -> bool:
            """兼容下载历史数据接口，返回 True 表示调用成功。"""
            self.logger.debug(
                "download_history_data(%s, period=%s, start=%s, end=%s)",
                stock_code,
                period,
                start_time,
                end_time,
            )
            self._fetch_stock_dataframe(
                stock_code, period, start_time=start_time, end_time=end_time, count=-1
            )
            return True

        def download_history_data2(
            self,
            stock_code_list: Sequence[str],
            period: str = "1d",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            incrementally: bool = True,
            callback: Optional[Any] = None,
            **_: Any,
        ) -> bool:
            """批量兼容接口，模拟进度回调。"""
            total = len(stock_code_list)
            for index, code in enumerate(stock_code_list, start=1):
                self.download_history_data(
                    code,
                    period=period,
                    start_time=start_time,
                    end_time=end_time,
                )
                if callback:
                    try:
                        callback({"finished": index, "total": total})
                    except Exception as exc:
                        self.logger.debug("进度回调执行失败: %s", exc)
            return True

        def get_market_data_ex(
            self,
            field_list: Optional[Iterable[str]] = None,
            stock_list: Optional[Sequence[str]] = None,
            period: str = "1d",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            count: int = -1,
            dividend_type: str = "none",
            **_: Any,
        ) -> Dict[str, pd.DataFrame]:
            """返回 {stock_code: DataFrame} 结构的行情数据。"""
            result: Dict[str, pd.DataFrame] = {}
            if not stock_list:
                return result

            for stock in stock_list:
                df = self._fetch_stock_dataframe(
                    stock,
                    period,
                    start_time=start_time,
                    end_time=end_time,
                    count=count,
                    dividend_type=dividend_type,
                )
                if df is None:
                    df = self._empty_dataframe(field_list)
                result[stock] = self._select_fields(df, field_list)
            return result

        def get_market_data(
            self,
            field_list: Optional[Iterable[str]] = None,
            stock_list: Optional[Sequence[str]] = None,
            period: str = "1d",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            count: int = -1,
            dividend_type: str = "none",
            **_: Any,
        ) -> Dict[str, pd.DataFrame]:
            """
            简化实现，返回 {field: DataFrame} 结构。
            若无法获取数据则返回空字典，调用方会回退至默认逻辑。
            """
            data_ex = self.get_market_data_ex(
                field_list=field_list,
                stock_list=stock_list,
                period=period,
                start_time=start_time,
                end_time=end_time,
                count=count,
                dividend_type=dividend_type,
            )
            if not data_ex:
                return {}

            if not field_list:
                return {}

            frames: Dict[str, pd.DataFrame] = {}
            for field in field_list:
                series_dict = {}
                index = None
                for stock, df in data_ex.items():
                    if field not in df.columns:
                        continue
                    series_dict[stock] = df[field]
                    if index is None and "time" in df.columns:
                        index = df["time"]
                if series_dict:
                    frames[field] = pd.DataFrame(series_dict, index=index)
            return frames

        def get_local_data(
            self,
            field_list: Optional[Iterable[str]] = None,
            stock_list: Optional[Sequence[str]] = None,
            period: str = "1d",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            count: int = -1,
            dividend_type: str = "none",
            **_: Any,
        ) -> Dict[str, pd.DataFrame]:
            """本地数据接口与 get_market_data_ex 保持一致。"""
            return self.get_market_data_ex(
                field_list=field_list,
                stock_list=stock_list,
                period=period,
                start_time=start_time,
                end_time=end_time,
                count=count,
                dividend_type=dividend_type,
            )

        def download_sector_data(self, **_: Any) -> bool:
            """板块数据下载占位，直接返回 True。"""
            return True

        def get_sector_list(self) -> List[str]:
            """返回本地已知板块列表。"""
            sectors = []
            for sector, file_name in self._SECTOR_FILE_MAP.items():
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    sectors.append(sector)
            return sectors

        def get_stock_list_in_sector(self, sector_name: str) -> List[str]:
            """读取本地 CSV 返回股票代码列表。"""
            file_path = os.path.join(
                self.data_dir, self._SECTOR_FILE_MAP.get(sector_name, "")
            )
            if not os.path.exists(file_path):
                return []
            try:
                df = pd.read_csv(file_path, header=None, names=["code", "name"])
                return df["code"].dropna().astype(str).tolist()
            except Exception as exc:
                self.logger.debug("读取板块文件失败 %s: %s", sector_name, exc)
                return []

        def get_instrument_detail(self, stock_code: str) -> Dict[str, Any]:
            """根据本地文件返回简单的合约信息。"""
            for file_name in self._SECTOR_FILE_MAP.values():
                file_path = os.path.join(self.data_dir, file_name)
                if not os.path.exists(file_path):
                    continue
                try:
                    df = pd.read_csv(file_path, header=None, names=["code", "name"])
                    matched = df[df["code"] == stock_code]
                    if not matched.empty:
                        name = matched.iloc[0]["name"]
                        return {"InstrumentID": stock_code, "InstrumentName": name}
                except Exception:
                    continue
            return {"InstrumentID": stock_code, "InstrumentName": stock_code}

        # --------- 内部工具方法 -----------
        def _empty_dataframe(
            self, field_list: Optional[Iterable[str]]
        ) -> pd.DataFrame:
            columns = ["time", "date"]
            if field_list:
                for field in field_list:
                    if field not in columns:
                        columns.append(field)
            return pd.DataFrame(columns=columns)

        def _fetch_stock_dataframe(
            self,
            stock: str,
            period: str,
            start_time: Optional[str],
            end_time: Optional[str],
            count: int,
            dividend_type: str = "none",
        ) -> Optional[pd.DataFrame]:
            if ak is None:
                return None

            if period in ("tick", "1s"):
                self.logger.debug("当前兼容实现不支持 %s 周期", period)
                return None

            try:
                if period == "1d":
                    df = ak.stock_zh_a_hist(
                        symbol=self._convert_symbol(stock, daily=True),
                        period="daily",
                        start_date=self._extract_date(start_time),
                        end_date=self._extract_date(end_time),
                        adjust=self._map_adjust(dividend_type),
                    )
                elif period in self._MINUTE_PERIOD_MAP:
                    df = ak.stock_zh_a_minute(
                        symbol=self._convert_symbol(stock, daily=False),
                        period=self._MINUTE_PERIOD_MAP[period],
                        adjust=self._map_adjust(dividend_type),
                    )

                    if start_time:
                        df = df[df["day"] >= self._extract_datetime_str(start_time)]
                    if end_time:
                        df = df[df["day"] <= self._extract_datetime_str(end_time)]
                else:
                    self.logger.debug("未实现的周期类型: %s", period)
                    return None
            except Exception as exc:
                self.logger.debug("akshare 获取 %s 数据失败: %s", stock, exc)
                return None

            if df is None or df.empty:
                return None

            df = self._standardize_dataframe(df, period)

            if start_time:
                start_dt = self._to_datetime(start_time)
                df = df[df["datetime"] >= start_dt]
            if end_time:
                end_dt = self._to_datetime(end_time)
                df = df[df["datetime"] <= end_dt]

            if count and count > 0:
                df = df.tail(count)

            return df.reset_index(drop=True)

        def _standardize_dataframe(
            self, df: pd.DataFrame, period: str
        ) -> pd.DataFrame:
            df = df.copy()
            if "日期" in df.columns:
                df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                    },
                    inplace=True,
                )
                df["datetime"] = pd.to_datetime(df["date"])
            elif "day" in df.columns:
                df.rename(
                    columns={
                        "day": "date",
                        "open": "open",
                        "close": "close",
                        "high": "high",
                        "low": "low",
                        "volume": "volume",
                        "amount": "amount",
                    },
                    inplace=True,
                )
                df["datetime"] = pd.to_datetime(df["date"])
            else:
                df["datetime"] = pd.to_datetime(df.index)

            df["time"] = (df["datetime"].view("int64") // 10**6).astype("int64")

            # 部分分钟数据没有“date”列，补全
            if "date" not in df.columns:
                df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")

            return df

        def _select_fields(
            self, df: pd.DataFrame, field_list: Optional[Iterable[str]]
        ) -> pd.DataFrame:
            if not field_list:
                return df
            columns = ["time", "date"]
            for field in field_list:
                if field not in columns and field in df.columns:
                    columns.append(field)
            return df.loc[:, [col for col in columns if col in df.columns]]

        @staticmethod
        def _convert_symbol(stock: str, *, daily: bool) -> str:
            if "." not in stock:
                return stock
            code, market = stock.split(".", 1)
            market = market.upper()
            if daily:
                return code
            prefix_map = {"SH": "sh", "SZ": "sz", "BJ": "bj"}
            return f"{prefix_map.get(market, '').lower()}{code}"

        @staticmethod
        def _map_adjust(dividend_type: str) -> str:
            mapping = {"pre": "qfq", "back": "hfq", "none": ""}
            return mapping.get(dividend_type or "none", "")

        @staticmethod
        def _extract_date(value: Optional[str]) -> str:
            if not value:
                return ""
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if len(digits) >= 8:
                return digits[:8]
            return ""

        @staticmethod
        def _extract_datetime_str(value: Optional[str]) -> str:
            if not value:
                return ""
            value = str(value)
            if len(value) >= 14 and value.isdigit():
                return (
                    f"{value[0:4]}-{value[4:6]}-{value[6:8]} "
                    f"{value[8:10]}:{value[10:12]}:{value[12:14]}"
                )
            if len(value) >= 8 and value.isdigit():
                return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"
            return value

        @staticmethod
        def _to_datetime(value: str) -> pd.Timestamp:
            text = str(value)
            if text.isdigit():
                if len(text) >= 14:
                    fmt = "%Y%m%d%H%M%S"
                elif len(text) >= 8:
                    fmt = "%Y%m%d"
                else:
                    fmt = "%Y"
                return pd.to_datetime(text[: len(fmt)], format=fmt)
            return pd.to_datetime(text)

    xtdata = _XtDataCompat()
    xtconstant = _XtConstantStub()
    StockAccount = _StockAccountStub

__all__ = [
    "xtdata",
    "XtQuantTrader",
    "XtQuantTraderCallback",
    "xtconstant",
    "StockAccount",
    "has_xtquant",
]
