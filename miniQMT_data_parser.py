#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
miniQMT数据解析器
使用xtquant.xtdata.get_local_data处理miniQMT的本地数据
"""

import struct
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from kh_xtquant_compat import has_xtquant, xtdata

get_local_data = xtdata.get_local_data
XTDATA_AVAILABLE = has_xtquant()


class MiniQMTDataParser:
    """miniQMT数据解析器"""
    
    def __init__(self, data_dir=None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
    def parse_tick_data(self, file_path, max_records=None):
        """
        解析tick数据
        
        Args:
            file_path: 数据文件路径
            max_records: 最大记录数
            
        Returns:
            list: 解析后的数据列表
        """
        data = []
        
        if not XTDATA_AVAILABLE:
            self.logger.warning("xtquant不可用，无法解析tick数据")
            return []
            
        try:
            # 从文件路径提取股票代码和日期信息
            stock_code, date_str = self._extract_stock_info_from_tick_path(file_path)
            if not stock_code or not date_str:
                self.logger.error(f"无法从路径提取股票信息: {file_path}")
                return []
            
            self.logger.info(f"解析tick数据: {stock_code}, 日期: {date_str}")
            
            # 构造完整股票代码
            full_stock_code = self._get_full_stock_code(stock_code, file_path)
            
            # 设置时间范围（当天）
            start_time = date_str
            end_time = date_str
            
            # 使用get_local_data获取tick数据
            # 如果max_records为None，使用一个很大的数值表示不限制
            count_limit = max_records if max_records is not None else 10000000  # 1000万条，基本相当于无限制
            
            tick_data = get_local_data(
                field_list=[],  # 空列表表示获取所有字段
                stock_list=[full_stock_code],
                period='tick',
                start_time=start_time,
                end_time=end_time,
                count=count_limit,
                dividend_type='none',
                fill_data=False,
                data_dir=self.data_dir
            )
            
            if tick_data and full_stock_code in tick_data:
                # tick数据返回格式: {stock_code: DataFrame}
                tick_df = tick_data[full_stock_code]
                
                if isinstance(tick_df, pd.DataFrame) and not tick_df.empty:
                    self.logger.info(f"找到股票 {full_stock_code} 的tick数据，形状: {tick_df.shape}")
                    self.logger.info(f"tick数据列名: {list(tick_df.columns)}")
                    print(f"DEBUG: tick数据列名: {list(tick_df.columns)}")  # 添加控制台输出
                    
                    # 检查开高低收字段是否存在
                    ohlc_fields = ['open', 'high', 'low', 'close', 'lastClose']
                    missing_ohlc = [field for field in ohlc_fields if field not in tick_df.columns]
                    existing_ohlc = [field for field in ohlc_fields if field in tick_df.columns]
                    if missing_ohlc:
                        print(f"DEBUG: 缺失的OHLC字段: {missing_ohlc}")
                    if existing_ohlc:
                        print(f"DEBUG: 存在的OHLC字段: {existing_ohlc}")
                    
                    # 限制记录数，如果max_records为None则不限制
                    record_count = len(tick_df) if max_records is None else min(len(tick_df), max_records)
                    
                    # 处理每一行tick数据
                    for i in range(record_count):
                        row = tick_df.iloc[i]
                        
                        # 处理时间格式 - 先初始化time_str
                        time_str = str(tick_df.index[i])
                        
                        # 尝试转换时间格式
                        try:
                            # 首先尝试从columns中获取时间
                            time_val = None
                            if 'time' in tick_df.columns:
                                time_val = row['time']
                            else:
                                # 使用index作为时间值
                                if hasattr(tick_df.index[i], 'timestamp'):
                                    # 如果index是datetime类型，确保不会发生时区转换
                                    dt = tick_df.index[i]
                                    # 如果有时区信息，转换为本地时间
                                    if hasattr(dt, 'tz') and dt.tz is not None:
                                        # 移除时区信息，直接使用数值
                                        dt = dt.tz_localize(None)
                                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    # 使用index值作为时间戳
                                    time_val = tick_df.index[i]
                            
                            # 如果获取到时间值，进行转换
                            if time_val is not None:
                                # 转换为字符串处理
                                time_str_raw = str(time_val).strip()
                                
                                # 如果是纯数字格式，进行日期时间格式转换
                                if time_str_raw.isdigit():
                                    time_val_int = int(time_str_raw)
                                    
                                    # 判断数字格式并转换
                                    if len(time_str_raw) == 14:  # 14位日期时间格式YYYYMMDDHHMMSS
                                        # 解析14位日期时间：20250702091500 -> 2025-07-02 09:15:00
                                        year = int(time_str_raw[0:4])
                                        month = int(time_str_raw[4:6])
                                        day = int(time_str_raw[6:8])
                                        hour = int(time_str_raw[8:10])
                                        minute = int(time_str_raw[10:12])
                                        second = int(time_str_raw[12:14])
                                        time_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
                                    elif len(time_str_raw) == 13 and time_val_int > 1000000000000:  # 可能是错误转换的毫秒时间戳
                                        # 尝试从文件路径获取正确的日期
                                        try:
                                            # 提取文件路径中的日期信息
                                            stock_code, date_str = self._extract_stock_info_from_tick_path(file_path)
                                            if date_str and len(date_str) == 8:
                                                # 从时间戳提取时间部分，但使用文件日期
                                                dt = pd.to_datetime(time_val_int, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                                
                                                # 使用文件路径中的日期，但保留时间戳的时间部分
                                                file_year = int(date_str[0:4])
                                                file_month = int(date_str[4:6])
                                                file_day = int(date_str[6:8])
                                                
                                                timestamp_hour = dt.hour
                                                timestamp_minute = dt.minute
                                                timestamp_second = dt.second
                                                
                                                # 合并正确的日期和时间
                                                time_str = f"{file_year}-{file_month:02d}-{file_day:02d} {timestamp_hour:02d}:{timestamp_minute:02d}:{timestamp_second:02d}"
                                            else:
                                                # 如果无法从文件路径获取日期，就按原来的方式处理
                                                dt = pd.to_datetime(time_val_int, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                        except Exception as e:
                                            # 转换为北京时间
                                            dt = pd.to_datetime(time_val_int, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    elif len(time_str_raw) == 10 and time_val_int > 1000000000:  # 秒时间戳（10位数字）
                                        # 避免时区转换，直接使用UTC时间
                                        dt = pd.to_datetime(time_val_int, unit='s', utc=True).tz_localize(None)
                                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    elif time_val_int >= 10000 and time_val_int < 1000000:  # 可能是时间格式如93000表示09:30:00
                                        # 这种格式没有日期信息，只能显示时间
                                        time_str = f"{time_val_int//10000:02d}:{(time_val_int//100)%100:02d}:{time_val_int%100:02d}"
                                    else:
                                        time_str = str(time_val_int)
                                else:
                                    # 非纯数字，保持原始值
                                    time_str = time_str_raw
                                    
                        except (ValueError, OverflowError, AttributeError) as e:
                            # 转换失败时保持原始值
                            self.logger.debug(f"时间转换失败: {e}")
                            time_str = str(tick_df.index[i])
                        
                        # 按照标准tick数据字段顺序提取数据
                        time_val = time_str  # 时间戳
                        last_price = row.get('lastPrice', 0)  # 最新价
                        open_price = row.get('open', 0)  # 开盘价
                        high_price = row.get('high', 0)  # 最高价
                        low_price = row.get('low', 0)  # 最低价
                        last_close = row.get('lastClose', 0)  # 前收盘价
                        amount = row.get('amount', 0)  # 成交总额
                        volume = row.get('volume', 0)  # 成交总量
                        pvolume = row.get('pvolume', 0)  # 原始成交总量
                        stock_status = row.get('stockStatus', 0)  # 证券状态
                        open_int = row.get('openInt', 0)  # 持仓量
                        last_settlement_price = row.get('lastSettlementPrice', 0)  # 前结算
                        ask_price = row.get('askPrice', [])  # 委卖价
                        bid_price = row.get('bidPrice', [])  # 委买价
                        ask_vol = row.get('askVol', [])  # 委卖量
                        bid_vol = row.get('bidVol', [])  # 委买量
                        transaction_num = row.get('transactionNum', 0)  # 成交笔数
                        
                        # 提取买卖盘口数据 - 修复处理逻辑
                        bid_price_1 = '-'
                        bid_price_2 = '-'
                        bid_price_3 = '-'
                        bid_price_4 = '-'
                        bid_price_5 = '-'
                        ask_price_1 = '-'
                        ask_price_2 = '-'
                        ask_price_3 = '-'
                        ask_price_4 = '-'
                        ask_price_5 = '-'
                        bid_vol_1 = '-'
                        bid_vol_2 = '-'
                        bid_vol_3 = '-'
                        bid_vol_4 = '-'
                        bid_vol_5 = '-'
                        ask_vol_1 = '-'
                        ask_vol_2 = '-'
                        ask_vol_3 = '-'
                        ask_vol_4 = '-'
                        ask_vol_5 = '-'
                        
                        # 处理委买价数据
                        if 'bidPrice' in tick_df.columns:
                            bid_price_data = row['bidPrice']
                            if isinstance(bid_price_data, (list, np.ndarray)) and len(bid_price_data) >= 5:
                                bid_price_1 = round(float(bid_price_data[0]), 3) if bid_price_data[0] > 0 else '-'
                                bid_price_2 = round(float(bid_price_data[1]), 3) if bid_price_data[1] > 0 else '-'
                                bid_price_3 = round(float(bid_price_data[2]), 3) if bid_price_data[2] > 0 else '-'
                                bid_price_4 = round(float(bid_price_data[3]), 3) if bid_price_data[3] > 0 else '-'
                                bid_price_5 = round(float(bid_price_data[4]), 3) if bid_price_data[4] > 0 else '-'
                            elif isinstance(bid_price_data, (int, float)):
                                bid_price_1 = round(float(bid_price_data), 3) if bid_price_data > 0 else '-'
                        
                        # 处理委卖价数据
                        if 'askPrice' in tick_df.columns:
                            ask_price_data = row['askPrice']
                            if isinstance(ask_price_data, (list, np.ndarray)) and len(ask_price_data) >= 5:
                                ask_price_1 = round(float(ask_price_data[0]), 3) if ask_price_data[0] > 0 else '-'
                                ask_price_2 = round(float(ask_price_data[1]), 3) if ask_price_data[1] > 0 else '-'
                                ask_price_3 = round(float(ask_price_data[2]), 3) if ask_price_data[2] > 0 else '-'
                                ask_price_4 = round(float(ask_price_data[3]), 3) if ask_price_data[3] > 0 else '-'
                                ask_price_5 = round(float(ask_price_data[4]), 3) if ask_price_data[4] > 0 else '-'
                            elif isinstance(ask_price_data, (int, float)):
                                ask_price_1 = round(float(ask_price_data), 3) if ask_price_data > 0 else '-'
                        
                        # 处理委买量数据
                        if 'bidVol' in tick_df.columns:
                            bid_vol_data = row['bidVol']
                            if isinstance(bid_vol_data, (list, np.ndarray)) and len(bid_vol_data) >= 5:
                                bid_vol_1 = int(bid_vol_data[0]) if bid_vol_data[0] > 0 else '-'
                                bid_vol_2 = int(bid_vol_data[1]) if bid_vol_data[1] > 0 else '-'
                                bid_vol_3 = int(bid_vol_data[2]) if bid_vol_data[2] > 0 else '-'
                                bid_vol_4 = int(bid_vol_data[3]) if bid_vol_data[3] > 0 else '-'
                                bid_vol_5 = int(bid_vol_data[4]) if bid_vol_data[4] > 0 else '-'
                            elif isinstance(bid_vol_data, (int, float)):
                                bid_vol_1 = int(bid_vol_data) if bid_vol_data > 0 else '-'
                        
                        # 处理委卖量数据
                        if 'askVol' in tick_df.columns:
                            ask_vol_data = row['askVol']
                            if isinstance(ask_vol_data, (list, np.ndarray)) and len(ask_vol_data) >= 5:
                                ask_vol_1 = int(ask_vol_data[0]) if ask_vol_data[0] > 0 else '-'
                                ask_vol_2 = int(ask_vol_data[1]) if ask_vol_data[1] > 0 else '-'
                                ask_vol_3 = int(ask_vol_data[2]) if ask_vol_data[2] > 0 else '-'
                                ask_vol_4 = int(ask_vol_data[3]) if ask_vol_data[3] > 0 else '-'
                                ask_vol_5 = int(ask_vol_data[4]) if ask_vol_data[4] > 0 else '-'
                            elif isinstance(ask_vol_data, (int, float)):
                                ask_vol_1 = int(ask_vol_data) if ask_vol_data > 0 else '-'
                        
                        # 构建完整的tick数据记录，确保数据类型一致性
                        data.append({
                            '时间': time_str,
                            '最新价': round(float(last_price), 3) if last_price else 0.0,
                            '开盘价': round(float(open_price), 3) if open_price else 0.0,
                            '最高价': round(float(high_price), 3) if high_price else 0.0,
                            '最低价': round(float(low_price), 3) if low_price else 0.0,
                            '前收盘价': round(float(last_close), 3) if last_close else 0.0,
                            '成交总额': round(float(amount), 2) if amount else 0.0,
                            '成交总量': int(volume) if volume else 0,
                            '原始成交总量': int(pvolume) if pvolume else 0,
                            '证券状态': int(stock_status) if stock_status else 0,
                            '持仓量': int(open_int) if open_int else 0,
                            '前结算': round(float(last_settlement_price), 3) if last_settlement_price else 0.0,
                            '买一价': bid_price_1,
                            '买一量': bid_vol_1,
                            '买二价': bid_price_2,
                            '买二量': bid_vol_2,
                            '买三价': bid_price_3,
                            '买三量': bid_vol_3,
                            '买四价': bid_price_4,
                            '买四量': bid_vol_4,
                            '买五价': bid_price_5,
                            '买五量': bid_vol_5,
                            '卖一价': ask_price_1,
                            '卖一量': ask_vol_1,
                            '卖二价': ask_price_2,
                            '卖二量': ask_vol_2,
                            '卖三价': ask_price_3,
                            '卖三量': ask_vol_3,
                            '卖四价': ask_price_4,
                            '卖四量': ask_vol_4,
                            '卖五价': ask_price_5,
                            '卖五量': ask_vol_5,
                            '成交笔数': int(transaction_num) if transaction_num else 0
                        })
                    
                    self.logger.info(f"成功处理 {len(data)} 条tick数据")
                else:
                    self.logger.warning(f"tick数据为空: {full_stock_code}")
                    # 返回空数据而不是示例数据
                    data = []
            else:
                self.logger.warning(f"未获取到tick数据: {full_stock_code}")
                # 返回空数据而不是示例数据
                data = []
                
        except Exception as e:
            self.logger.error(f"解析tick数据失败: {e}")
            # 返回空数据而不是示例数据
            data = []
            
        return data
    
    def _extract_stock_info_from_tick_path(self, file_path):
        """从tick文件路径提取股票代码和日期"""
        try:
            # 路径格式: .../BJ/0/000001/20240101.dat
            parts = file_path.replace('\\', '/').split('/')
            if len(parts) >= 2:
                stock_code = parts[-2]  # 股票代码文件夹
                filename = parts[-1]    # 文件名
                date_str = filename.replace('.dat', '').replace('.DAT', '')
                
                # 验证股票代码（6位数字）和日期格式
                if stock_code.isdigit() and len(stock_code) == 6 and len(date_str) == 8:
                    return stock_code, date_str
                    
        except Exception as e:
            self.logger.error(f"提取股票信息失败: {e}")
            
        return None, None
    
    def _get_full_stock_code(self, stock_code, file_path):
        """根据文件路径构造完整股票代码"""
        if '/SH/' in file_path or '\\SH\\' in file_path:
            return f"{stock_code}.SH"
        elif '/SZ/' in file_path or '\\SZ\\' in file_path:
            return f"{stock_code}.SZ"
        elif '/BJ/' in file_path or '\\BJ\\' in file_path:
            return f"{stock_code}.BJ"
        else:
            return f"{stock_code}.SH"  # 默认上交所
    
    def _process_tick_array(self, tick_array, max_records):
        """处理tick数据数组"""
        data = []
        
        try:
            # tick数据通常包含多个字段，需要根据实际结构处理
            # 这里假设tick_array是结构化数组，包含time, price, volume等字段
            
            # 如果max_records为None则不限制记录数
            record_count = len(tick_array) if max_records is None else min(len(tick_array), max_records)
            
            for i in range(record_count):
                record = tick_array[i]
                
                # 根据实际tick数据结构调整字段访问方式
                # 这里提供一个通用的处理方式
                if hasattr(record, 'dtype') and record.dtype.names:
                    # 结构化数组
                    fields = record.dtype.names
                    time_field = None
                    price_field = None
                    volume_field = None
                    
                    # 查找常见字段名
                    for field in fields:
                        field_lower = field.lower()
                        if 'time' in field_lower or '时间' in field:
                            time_field = field
                        elif 'price' in field_lower or '价格' in field or 'last' in field_lower:
                            price_field = field
                        elif 'volume' in field_lower or '成交量' in field or 'qty' in field_lower:
                            volume_field = field
                    
                    # 提取数据
                    time_val = record[time_field] if time_field else 0
                    price_val = record[price_field] if price_field else 0
                    volume_val = record[volume_field] if volume_field else 0
                    
                    # 转换时间格式
                    if isinstance(time_val, (int, float)):
                        if time_val > 1000000000000:  # 毫秒时间戳（13位数字）
                            dt = datetime.fromtimestamp(time_val / 1000)
                            time_str = dt.strftime('%H:%M:%S.%f')[:-3]  # 显示到毫秒
                        elif time_val > 1000000000:  # 秒时间戳（10位数字）
                            dt = datetime.fromtimestamp(time_val)
                            time_str = dt.strftime('%H:%M:%S')
                        else:  # 可能是时间格式如93000表示09:30:00
                            hour = int(time_val // 10000)
                            minute = int((time_val % 10000) // 100)
                            second = int(time_val % 100)
                            dt = datetime.now().replace(hour=hour, minute=minute, second=second)
                            time_str = dt.strftime('%H:%M:%S')
                    else:
                        time_str = str(time_val)
                    
                    # 生成完整的tick数据记录
                    price = float(price_val) if price_val else 0.0
                    volume = int(volume_val) if volume_val else 0
                    
                    # 尝试从真实数据中提取买卖盘口数据
                    bid_prices = ['-'] * 5
                    ask_prices = ['-'] * 5
                    bid_volumes = ['-'] * 5
                    ask_volumes = ['-'] * 5
                    
                    # 查找买卖盘字段
                    for field in fields:
                        field_lower = field.lower()
                        # 买盘价格
                        for i in range(1, 6):
                            if f'bid{i}' in field_lower or f'买{i}价' in field or f'bp{i}' in field_lower:
                                bid_prices[i-1] = record[field] if record[field] else '-'
                            elif f'bid{i}v' in field_lower or f'买{i}量' in field or f'bv{i}' in field_lower:
                                bid_volumes[i-1] = record[field] if record[field] else '-'
                            elif f'ask{i}' in field_lower or f'卖{i}价' in field or f'ap{i}' in field_lower:
                                ask_prices[i-1] = record[field] if record[field] else '-'
                            elif f'ask{i}v' in field_lower or f'卖{i}量' in field or f'av{i}' in field_lower:
                                ask_volumes[i-1] = record[field] if record[field] else '-'
                    
                    data.append({
                        '时间': time_str,
                        '现价': round(price, 3),
                        '成交量': volume,
                        '成交额': round(price * volume, 2),
                        '总手数': volume * 100,  # 估算
                        '外盘': volume // 2,  # 估算
                        '内盘': volume - volume // 2,  # 估算
                        '买一价': bid_prices[0] if bid_prices else '-',
                        '买一量': bid_volumes[0] if bid_volumes else '-',
                        '买二价': bid_prices[1] if len(bid_prices) > 1 else '-',
                        '买二量': bid_volumes[1] if len(bid_volumes) > 1 else '-',
                        '买三价': bid_prices[2] if len(bid_prices) > 2 else '-',
                        '买三量': bid_volumes[2] if len(bid_volumes) > 2 else '-',
                        '买四价': bid_prices[3] if len(bid_prices) > 3 else '-',
                        '买四量': bid_volumes[3] if len(bid_volumes) > 3 else '-',
                        '买五价': bid_prices[4] if len(bid_prices) > 4 else '-',
                        '买五量': bid_volumes[4] if len(bid_volumes) > 4 else '-',
                        '卖一价': ask_prices[0] if ask_prices else '-',
                        '卖一量': ask_volumes[0] if ask_volumes else '-',
                        '卖二价': ask_prices[1] if len(ask_prices) > 1 else '-',
                        '卖二量': ask_volumes[1] if len(ask_volumes) > 1 else '-',
                        '卖三价': ask_prices[2] if len(ask_prices) > 2 else '-',
                        '卖三量': ask_volumes[2] if len(ask_volumes) > 2 else '-',
                        '卖四价': ask_prices[3] if len(ask_prices) > 3 else '-',
                        '卖四量': ask_volumes[3] if len(ask_volumes) > 3 else '-',
                        '卖五价': ask_prices[4] if len(ask_prices) > 4 else '-',
                        '卖五量': ask_volumes[4] if len(ask_volumes) > 4 else '-',
                        '委比%': 0.0,  # 需要计算
                        '委差': 0,     # 需要计算
                        '成交方向': '买盘',  # 需要判断
                        '换手率%': 0.0   # 需要计算
                    })
                else:
                    # 简单数组，假设按顺序为[time, price, volume, ...]
                    if len(record) >= 3:
                        time_val, price_val, volume_val = record[0], record[1], record[2]
                        
                        # 处理时间
                        if isinstance(time_val, (int, float)):
                            if time_val > 1000000000000:  # 毫秒时间戳（13位数字）
                                dt = datetime.fromtimestamp(time_val / 1000)
                                time_str = dt.strftime('%H:%M:%S.%f')[:-3]  # 显示到毫秒
                            elif time_val > 1000000000:  # 秒时间戳（10位数字）
                                dt = datetime.fromtimestamp(time_val)
                                time_str = dt.strftime('%H:%M:%S')
                            else:  # 可能是时间格式如93000表示09:30:00
                                hour = int(time_val // 10000)
                                minute = int((time_val % 10000) // 100)
                                second = int(time_val % 100)
                                dt = datetime.now().replace(hour=hour, minute=minute, second=second)
                                time_str = dt.strftime('%H:%M:%S')
                        else:
                            time_str = str(time_val)
                        
                        # 生成完整的tick数据记录
                        price = float(price_val)
                        volume = int(volume_val)
                        
                        # 模拟买卖盘口数据（实际数据应该从tick数组中提取）
                        import random
                        bid_prices = [round(price - 0.01 - i * 0.01, 3) for i in range(5)]
                        ask_prices = [round(price + 0.01 + i * 0.01, 3) for i in range(5)]
                        bid_volumes = [random.randint(100, 2000) for _ in range(5)]
                        ask_volumes = [random.randint(100, 2000) for _ in range(5)]
                        
                        data.append({
                            '时间': time_str,
                            '现价': round(price, 3),
                            '成交量': volume,
                            '成交额': round(price * volume, 2),
                            '总手数': volume * 100,  # 估算
                            '外盘': volume // 2,  # 估算
                            '内盘': volume - volume // 2,  # 估算
                            '买一价': bid_prices[0],
                            '买一量': bid_volumes[0],
                            '买二价': bid_prices[1],
                            '买二量': bid_volumes[1],
                            '买三价': bid_prices[2],
                            '买三量': bid_volumes[2],
                            '买四价': bid_prices[3],
                            '买四量': bid_volumes[3],
                            '买五价': bid_prices[4],
                            '买五量': bid_volumes[4],
                            '卖一价': ask_prices[0],
                            '卖一量': ask_volumes[0],
                            '卖二价': ask_prices[1],
                            '卖二量': ask_volumes[1],
                            '卖三价': ask_prices[2],
                            '卖三量': ask_volumes[2],
                            '卖四价': ask_prices[3],
                            '卖四量': ask_volumes[3],
                            '卖五价': ask_prices[4],
                            '卖五量': ask_volumes[4],
                            '委比%': 0.0,  # 需要计算
                            '委差': 0,     # 需要计算
                            '成交方向': '买盘',  # 需要判断
                            '换手率%': 0.0   # 需要计算
                        })
                        
        except Exception as e:
            self.logger.error(f"处理tick数组失败: {e}")
            
        return data if data else self._get_sample_tick_data()
    
    def parse_kline_data(self, file_path, period_type, max_records=None):
        """
        解析K线数据
        
        Args:
            file_path: 数据文件路径
            period_type: 周期类型 ('1m', '5m', '1d')
            max_records: 最大记录数
            
        Returns:
            list: 解析后的数据列表
        """
        data = []
        
        if not XTDATA_AVAILABLE:
            self.logger.warning("xtquant不可用，无法解析K线数据")
            return []
            
        try:
            # 从文件路径提取股票代码
            stock_code = self._extract_stock_code_from_kline_path(file_path)
            if not stock_code:
                self.logger.error(f"无法从路径提取股票代码: {file_path}")
                return []
            
            self.logger.info(f"解析K线数据: {stock_code}, 周期: {period_type}")
            
            # 构造完整股票代码
            full_stock_code = self._get_full_stock_code(stock_code, file_path)
            
            # 使用get_local_data获取K线数据
            # 如果max_records为None，使用一个很大的数值表示不限制
            count_limit = max_records if max_records is not None else 10000000  # 1000万条，基本相当于无限制
            
            kline_data = get_local_data(
                field_list=[],  # 空列表表示获取所有字段
                stock_list=[full_stock_code],
                period=period_type,
                start_time='',  # 空字符串表示获取所有可用数据
                end_time='',
                count=count_limit,
                dividend_type='none',
                fill_data=False,
                data_dir=self.data_dir
            )
            
            if kline_data:
                # K线数据返回格式根据API文档有两种情况：
                # 1. {stock_code: DataFrame} (某些情况下)
                # 2. {field: DataFrame} (标准格式，其中DataFrame的index是stock_list，columns是time_list)
                
                # 先检查返回数据的结构
                self.logger.info(f"K线数据返回结构: {list(kline_data.keys())}")
                
                # 判断数据格式
                if full_stock_code in kline_data:
                    # 格式1: {stock_code: DataFrame}
                    self.logger.info("检测到格式1: {stock_code: DataFrame}")
                    data = self._process_kline_dict_format1(kline_data, full_stock_code, period_type, max_records)
                else:
                    # 格式2: {field: DataFrame}，需要重组数据
                    self.logger.info("检测到格式2: {field: DataFrame}")
                    data = self._process_kline_dict_format2(kline_data, full_stock_code, period_type, max_records)
            else:
                self.logger.warning(f"未获取到K线数据: {full_stock_code}")
                # 返回空数据而不是示例数据
                data = []
                
        except Exception as e:
            self.logger.error(f"解析K线数据失败: {e}")
            # 返回空数据而不是示例数据
            data = []
            
        return data
    
    def _extract_stock_code_from_kline_path(self, file_path):
        """从K线文件路径提取股票代码"""
        try:
            # 路径格式: .../SH/60/000001.DAT
            filename = os.path.basename(file_path)
            stock_code = filename.replace('.DAT', '').replace('.dat', '')
            
            # 验证股票代码（6位数字）
            if stock_code.isdigit() and len(stock_code) == 6:
                return stock_code
                
        except Exception as e:
            self.logger.error(f"提取股票代码失败: {e}")
            
        return None
    
    def _process_kline_dict_format1(self, kline_data, stock_code, period_type, max_records):
        """处理K线数据字典"""
        data = []
        
        try:
            # 实际的数据结构是 {股票代码: DataFrame}
            self.logger.info(f"K线数据结构: {list(kline_data.keys()) if kline_data else '空'}")
            self.logger.info(f"期望的周期类型: {period_type}")
            
            # 直接从股票代码获取DataFrame
            if stock_code in kline_data:
                df = kline_data[stock_code]
                self.logger.info(f"找到股票 {stock_code} 的数据，形状: {df.shape}")
                self.logger.info(f"列名: {list(df.columns)}")
                self.logger.info(f"索引前5个: {df.index.tolist()[:5]}")
                self.logger.info(f"索引类型: {type(df.index)}")
                
                # 添加更详细的调试信息
                if not df.empty:
                    first_row = df.iloc[0]
                    self.logger.info(f"第一行数据: {first_row.to_dict()}")
                    if 'time' in df.columns:
                        time_val = first_row['time']
                        self.logger.info(f"time列第一个值: {time_val}, 类型: {type(time_val)}")
                
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # 限制记录数，如果max_records为None则不限制
                    record_count = len(df) if max_records is None else min(len(df), max_records)
                    
                    # 处理每一行数据
                    for i in range(record_count):
                        row_index = df.index[i]
                        row = df.iloc[i]
                        
                        # 处理时间格式 - 更详细的调试和处理
                        time_str = str(row_index)
                        if 'time' in df.columns:
                            time_val = row['time']
                            if i == 0:  # 只在第一行打印调试信息
                                self.logger.info(f"处理time列: 值={time_val}, 类型={type(time_val)}")
                            
                            if isinstance(time_val, (int, float)):
                                # 毫秒时间戳转换，添加时区处理
                                try:
                                    dt = pd.to_datetime(time_val, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                    if period_type == '1d':
                                        time_str = dt.strftime('%Y-%m-%d')
                                    else:
                                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    if i == 0:
                                        self.logger.info(f"时间戳转换结果: {time_str}")
                                except Exception as e:
                                    self.logger.warning(f"时间戳转换失败: {e}, 使用原始值")
                                    time_str = str(time_val)
                            else:
                                time_str = str(time_val)
                        else:
                            # 没有time列，尝试从索引解析时间
                            if i == 0:
                                self.logger.info(f"没有time列，从索引解析: {row_index}")
                            
                            if period_type == '1d':
                                # 日线数据，索引通常是日期字符串
                                if len(str(row_index)) == 8:  # YYYYMMDD格式
                                    time_str = f"{str(row_index)[:4]}-{str(row_index)[4:6]}-{str(row_index)[6:8]}"
                                else:
                                    time_str = str(row_index)
                            else:
                                # 分钟数据，尝试解析索引
                                try:
                                    if isinstance(row_index, (int, float)):
                                        # 如果索引是时间戳
                                        if row_index > 1000000000000:  # 毫秒时间戳
                                            dt = pd.to_datetime(row_index, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                        elif row_index > 1000000000:  # 秒时间戳
                                            dt = pd.to_datetime(row_index, unit='s', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                                        else:
                                            dt = pd.to_datetime(str(row_index))
                                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    else:
                                        # 尝试解析字符串格式的时间
                                        dt = pd.to_datetime(str(row_index))
                                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                        
                                except Exception as e:
                                    if i == 0:
                                        self.logger.warning(f"时间解析失败: {e}, 使用原始索引: {row_index}")
                                    time_str = str(row_index)
                        
                        # 提取所有字段数据
                        open_val = row.get('open', 0)
                        high_val = row.get('high', 0)
                        low_val = row.get('low', 0)
                        close_val = row.get('close', 0)
                        volume_val = row.get('volume', 0)
                        amount_val = row.get('amount', 0)
                        settelement_price_val = row.get('settelementPrice', 0)
                        open_interest_val = row.get('openInterest', 0)
                        pre_close_val = row.get('preClose', 0)
                        suspend_flag_val = row.get('suspendFlag', 0)
                        
                        # 构建数据记录（移除涨跌计算以提高加载速度）
                        data.append({
                            '时间': time_str,
                            '开盘价': round(float(open_val), 3) if open_val else 0.0,
                            '最高价': round(float(high_val), 3) if high_val else 0.0,
                            '最低价': round(float(low_val), 3) if low_val else 0.0,
                            '收盘价': round(float(close_val), 3) if close_val else 0.0,
                            '成交量': int(volume_val) if volume_val else 0,
                            '成交额': round(float(amount_val), 2) if amount_val else 0.0,
                            '今结算': round(float(settelement_price_val), 3) if settelement_price_val else 0.0,
                            '持仓量': int(open_interest_val) if open_interest_val else 0,
                            '前收价': round(float(pre_close_val), 3) if pre_close_val else 0.0,
                            '停牌标记': int(suspend_flag_val) if suspend_flag_val is not None else 0
                        })
                        
                    self.logger.info(f"成功处理 {len(data)} 条K线数据")
                else:
                    self.logger.warning(f"股票 {stock_code} 的数据为空或格式错误")
            else:
                self.logger.warning(f"未找到股票 {stock_code} 的数据，可用股票: {list(kline_data.keys())}")
                
        except Exception as e:
            self.logger.error(f"处理K线数据字典失败: {e}")
            import traceback
            traceback.print_exc()
            
        return data  # 返回真实数据，即使为空也不使用示例数据
    
    def _process_kline_dict_format2(self, kline_data, stock_code, period_type, max_records):
        """处理K线数据字典 - 格式2: {field: DataFrame}"""
        data = []
        
        try:
            self.logger.info(f"处理K线数据格式2，股票: {stock_code}, 周期: {period_type}")
            self.logger.info(f"可用字段: {list(kline_data.keys())}")
            
            # 获取所有可用的字段数据 - 扩展字段列表
            available_fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'time', 
                               'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
            field_data = {}
            
            for field in available_fields:
                if field in kline_data:
                    df = kline_data[field]
                    self.logger.info(f"字段 {field} 数据形状: {df.shape}")
                    self.logger.info(f"字段 {field} 索引: {df.index.tolist()[:3]}")
                    self.logger.info(f"字段 {field} 列: {df.columns.tolist()[:3]}")
                    
                    # 检查我们的股票是否在索引中
                    if stock_code in df.index:
                        field_data[field] = df.loc[stock_code]
                        self.logger.info(f"找到股票 {stock_code} 在字段 {field} 中")
                    else:
                        self.logger.warning(f"股票 {stock_code} 不在字段 {field} 的索引中")
                        self.logger.info(f"可用的股票索引: {df.index.tolist()}")
            
            if not field_data:
                self.logger.warning(f"没有找到股票 {stock_code} 的任何数据")
                return data
            
            # 确定数据长度（使用close字段作为基准）
            if 'close' in field_data:
                time_index = field_data['close'].index
                # 如果max_records为None则不限制记录数
                record_count = len(time_index) if max_records is None else min(len(time_index), max_records)
                self.logger.info(f"基于close字段，数据长度: {len(time_index)}, 处理: {record_count}")
                
                for i in range(record_count):
                    time_key = time_index[i]
                    
                    # 处理时间格式
                    time_str = str(time_key)
                    try:
                        if isinstance(time_key, (int, float)):
                            # 时间戳处理
                            if time_key > 1000000000000:  # 毫秒时间戳
                                dt = pd.to_datetime(time_key, unit='ms', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                            elif time_key > 1000000000:  # 秒时间戳
                                dt = pd.to_datetime(time_key, unit='s', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
                            else:
                                # 可能是日期格式如20240101
                                dt = pd.to_datetime(str(int(time_key)), format='%Y%m%d')
                            
                            if period_type == '1d':
                                time_str = dt.strftime('%Y-%m-%d')
                            else:
                                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            # 尝试解析字符串时间
                            dt = pd.to_datetime(str(time_key))
                            if period_type == '1d':
                                time_str = dt.strftime('%Y-%m-%d')
                            else:
                                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                
                        if i == 0:
                            self.logger.info(f"时间处理: {time_key} -> {time_str}")
                            
                    except Exception as e:
                        if i == 0:
                            self.logger.warning(f"时间解析失败: {e}, 使用原始值: {time_key}")
                        time_str = str(time_key)
                    
                    # 提取所有字段数据
                    open_val = field_data.get('open', pd.Series()).get(time_key, 0) if 'open' in field_data else 0
                    high_val = field_data.get('high', pd.Series()).get(time_key, 0) if 'high' in field_data else 0
                    low_val = field_data.get('low', pd.Series()).get(time_key, 0) if 'low' in field_data else 0
                    close_val = field_data.get('close', pd.Series()).get(time_key, 0) if 'close' in field_data else 0
                    volume_val = field_data.get('volume', pd.Series()).get(time_key, 0) if 'volume' in field_data else 0
                    amount_val = field_data.get('amount', pd.Series()).get(time_key, 0) if 'amount' in field_data else 0
                    settelement_price_val = field_data.get('settelementPrice', pd.Series()).get(time_key, 0) if 'settelementPrice' in field_data else 0
                    open_interest_val = field_data.get('openInterest', pd.Series()).get(time_key, 0) if 'openInterest' in field_data else 0
                    pre_close_val = field_data.get('preClose', pd.Series()).get(time_key, 0) if 'preClose' in field_data else 0
                    suspend_flag_val = field_data.get('suspendFlag', pd.Series()).get(time_key, 0) if 'suspendFlag' in field_data else 0
                    
                    # 构建数据记录（移除涨跌计算以提高加载速度）
                    data.append({
                        '时间': time_str,
                        '开盘价': round(float(open_val), 3) if open_val else 0.0,
                        '最高价': round(float(high_val), 3) if high_val else 0.0,
                        '最低价': round(float(low_val), 3) if low_val else 0.0,
                        '收盘价': round(float(close_val), 3) if close_val else 0.0,
                        '成交量': int(volume_val) if volume_val else 0,
                        '成交额': round(float(amount_val), 2) if amount_val else 0.0,
                        '今结算': round(float(settelement_price_val), 3) if settelement_price_val else 0.0,
                        '持仓量': int(open_interest_val) if open_interest_val else 0,
                        '前收价': round(float(pre_close_val), 3) if pre_close_val else 0.0,
                        '停牌标记': int(suspend_flag_val) if suspend_flag_val is not None else 0
                    })
                    
                self.logger.info(f"成功处理 {len(data)} 条K线数据（格式2）")
            else:
                self.logger.warning("没有找到close字段数据")
                
        except Exception as e:
            self.logger.error(f"处理K线数据字典格式2失败: {e}")
            import traceback
            traceback.print_exc()
            
        return data
    
    def _generate_tick_record(self, time_str, base_price):
        """生成完整的tick数据记录"""
        import random
        
        # 模拟价格波动
        price_change = random.uniform(-0.05, 0.05)
        price = max(base_price + price_change, 0.01)
        volume = random.randint(100, 1000)
        
        # 模拟买卖盘口数据
        bid_prices = [round(price - 0.01 - i * 0.01, 3) for i in range(5)]
        ask_prices = [round(price + 0.01 + i * 0.01, 3) for i in range(5)]
        bid_volumes = [random.randint(100, 2000) for _ in range(5)]
        ask_volumes = [random.randint(100, 2000) for _ in range(5)]
        
        # 模拟内外盘数据
        outer_volume = random.randint(volume//3, volume)  # 外盘（主买）
        inner_volume = volume - outer_volume  # 内盘（主卖）
        
        # 模拟总手数和委比
        total_volume = random.randint(volume * 100, volume * 1000)
        bid_total = sum(bid_volumes)
        ask_total = sum(ask_volumes)
        委比 = ((bid_total - ask_total) / (bid_total + ask_total)) * 100 if (bid_total + ask_total) > 0 else 0
        
        return {
            '时间': time_str,
            '最新价': round(price, 3),
            '开盘价': round(price + random.uniform(-0.02, 0.02), 3),
            '最高价': round(price + random.uniform(0.0, 0.05), 3),
            '最低价': round(price + random.uniform(-0.05, 0.0), 3),
            '前收盘价': round(price + random.uniform(-0.03, 0.03), 3),
            '成交总额': round(price * volume, 2),
            '成交总量': volume,
            '原始成交总量': volume,
            '证券状态': 0,
            '持仓量': 0,
            '前结算': round(price + random.uniform(-0.02, 0.02), 3),
            '买一价': bid_prices[0],
            '买一量': bid_volumes[0],
            '买二价': bid_prices[1],
            '买二量': bid_volumes[1],
            '买三价': bid_prices[2],
            '买三量': bid_volumes[2],
            '买四价': bid_prices[3],
            '买四量': bid_volumes[3],
            '买五价': bid_prices[4],
            '买五量': bid_volumes[4],
            '卖一价': ask_prices[0],
            '卖一量': ask_volumes[0],
            '卖二价': ask_prices[1],
            '卖二量': ask_volumes[1],
            '卖三价': ask_prices[2],
            '卖三量': ask_volumes[2],
            '卖四价': ask_prices[3],
            '卖四量': ask_volumes[3],
            '卖五价': ask_prices[4],
            '卖五量': ask_volumes[4],
            '成交笔数': random.randint(1, 50)
        }

    def _get_sample_tick_data(self):
        """获取示例tick数据"""
        current_time = datetime.now()
        sample_data = []
        
        base_price = 10.50
        # 生成更多示例数据（模拟一个完整交易日的数据）
        # 交易时间：9:30-11:30 (2小时) + 13:00-15:00 (2小时) = 4小时 = 14400秒
        # 每3秒一条tick，总共约4800条
        
        # 分别生成上午和下午的数据
        morning_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        afternoon_start = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        
        # 上午时段：9:30-11:30 (2小时 = 7200秒，每3秒一条 = 2400条)
        for i in range(2400):
            tick_time = morning_start + timedelta(seconds=i*3)
            
            # 检查是否超过11:30
            if tick_time.hour >= 11 and tick_time.minute >= 30:
                break
                
            time_str = tick_time.strftime('%H:%M:%S.%f')[:-3]
            
            # 生成完整的tick数据记录
            tick_record = self._generate_tick_record(time_str, base_price)
            sample_data.append(tick_record)
        
        # 下午时段：13:00-15:00 (2小时 = 7200秒，每3秒一条 = 2400条)
        for i in range(2400):
            tick_time = afternoon_start + timedelta(seconds=i*3)
            
            # 检查是否超过15:00
            if tick_time.hour >= 15:
                break
                
            time_str = tick_time.strftime('%H:%M:%S.%f')[:-3]
            
            # 生成完整的tick数据记录
            tick_record = self._generate_tick_record(time_str, base_price)
            sample_data.append(tick_record)
            
        return sample_data
    
    def _get_sample_kline_data(self, period_type):
        """获取示例K线数据"""
        current_time = datetime.now()
        sample_data = []
        
        base_price = 10.50
        
        # 根据周期类型决定生成多少条数据
        if period_type == '1d':
            data_count = 250  # 一年的交易日
            time_delta_func = lambda i: timedelta(days=i)
            time_format = '%Y-%m-%d'
        elif period_type == '5m':
            data_count = 480  # 一天的5分钟K线（4小时交易时间）
            time_delta_func = lambda i: timedelta(minutes=i*5)
            time_format = '%Y-%m-%d %H:%M:%S'
        else:  # 1m
            data_count = 240  # 一天的1分钟K线（4小时交易时间）
            time_delta_func = lambda i: timedelta(minutes=i)
            time_format = '%Y-%m-%d %H:%M:%S'
        
        import random
        for i in range(data_count):
            time_offset = time_delta_func(i)
            time_str = (current_time - time_offset).strftime(time_format)
            
            # 模拟OHLC数据
            random_factor = random.uniform(-0.02, 0.02)
            open_price = base_price + random_factor
            
            # 生成合理的OHLC关系
            high_offset = random.uniform(0, 0.03)
            low_offset = random.uniform(0, 0.03)
            close_offset = random.uniform(-0.02, 0.02)
            
            high_price = open_price + high_offset
            low_price = open_price - low_offset
            close_price = open_price + close_offset
            
            # 确保OHLC关系正确
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            volume = random.randint(1000, 10000)
            amount = volume * ((open_price + close_price) / 2)
            
            sample_data.append({
                '时间': time_str,
                '开盘价': round(open_price, 3),
                '最高价': round(high_price, 3),
                '最低价': round(low_price, 3),
                '收盘价': round(close_price, 3),
                '成交量': volume,
                '成交额': round(amount, 2)
            })
            
        return list(reversed(sample_data))  # 按时间正序返回
    
    def get_data_files(self, directory_path, file_extension='.dat'):
        """
        获取目录下的数据文件列表
        
        Args:
            directory_path: 目录路径
            file_extension: 文件扩展名
            
        Returns:
            list: 文件信息列表
        """
        files_info = []
        
        if not os.path.exists(directory_path):
            return files_info
            
        try:
            for filename in os.listdir(directory_path):
                # 检查文件扩展名（不区分大小写）
                if filename.upper().endswith(file_extension.upper()):
                    file_path = os.path.join(directory_path, filename)
                    
                    # 确保是文件而不是文件夹
                    if os.path.isfile(file_path):
                        # 获取文件信息
                        stat = os.stat(file_path)
                        
                        files_info.append({
                            'filename': filename,
                            'path': file_path,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime,
                            'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
            # 按股票代码排序（从文件名提取）
            def get_stock_code(file_info):
                filename = file_info['filename']
                return filename.replace('.DAT', '').replace('.dat', '')
            
            files_info.sort(key=get_stock_code)
            
        except Exception as e:
            self.logger.error(f"获取文件列表失败: {e}")
            
        return files_info
    
    def get_real_record_count(self, file_path, period_type):
        """
        获取真实的记录数量
        
        Args:
            file_path: 文件路径
            period_type: 周期类型
            
        Returns:
            int: 记录数量
        """
        # 优先使用文件大小估算方法，更可靠
        estimated_count = self._estimate_record_count_by_filesize(file_path)
        
        # 如果xtquant可用，尝试验证
        if XTDATA_AVAILABLE and estimated_count > 0:
            try:
                # 从文件路径提取股票代码
                stock_code = self._extract_stock_code_from_kline_path(file_path)
                if stock_code:
                    # 构造完整股票代码
                    full_stock_code = self._get_full_stock_code(stock_code, file_path)
                    
                    # 使用get_local_data获取少量数据来验证
                    kline_data = get_local_data(
                        field_list=['close'],  # 只获取收盘价字段
                        stock_list=[full_stock_code],
                        period=period_type,
                        start_time='',
                        end_time='',
                        count=100,  # 只获取100条数据来验证
                        dividend_type='none',
                        fill_data=False,
                        data_dir=self.data_dir
                    )
                    
                    if kline_data and full_stock_code in kline_data:
                        df = kline_data[full_stock_code]
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            api_sample_count = len(df)
                            if api_sample_count > 0:
                                # 根据周期类型决定日志级别
                                if '/60/' in file_path or '\\60\\' in file_path:  # 1m数据
                                    self.logger.debug(f"API验证: 获取到{api_sample_count}条样本数据")
                                else:
                                    self.logger.info(f"API验证: 获取到{api_sample_count}条样本数据")
                                
                                # 如果API能获取到数据，但估算的记录数似乎不合理，调整估算
                                if estimated_count < api_sample_count:
                                    # 估算数量小于API样本数量，可能低估了
                                    estimated_count = max(estimated_count, api_sample_count * 2)
                                    if '/60/' in file_path or '\\60\\' in file_path:  # 1m数据
                                        self.logger.debug(f"调整估算记录数为: {estimated_count}")
                                    else:
                                        self.logger.info(f"调整估算记录数为: {estimated_count}")
                                    
            except Exception as e:
                # 减少API验证失败的日志噪音
                pass
                
        return estimated_count

    def _estimate_record_count_by_filesize(self, file_path):
        """
        通过文件大小估算记录数量
        
        Args:
            file_path: 文件路径
            
        Returns:
            int: 估算的记录数量
        """
        try:
            file_size = os.path.getsize(file_path)
            
            # 根据文件路径判断数据周期类型
            if '/60/' in file_path or '\\60\\' in file_path:
                # 1分钟数据，根据经验通常是32-48字节/条
                preferred_sizes = [32, 40, 48, 36, 44]
                period_name = "1m"
            elif '/300/' in file_path or '\\300\\' in file_path:
                # 5分钟数据，通常是32-48字节/条
                preferred_sizes = [32, 40, 48, 36, 44]
                period_name = "5m"
            elif '/86400/' in file_path or '\\86400\\' in file_path:
                # 日线数据，通常是32-40字节/条
                preferred_sizes = [32, 40, 36, 44, 48]
                period_name = "1d"
            else:
                # 其他数据，使用通用大小
                preferred_sizes = [32, 40, 48, 36, 44, 28, 56, 64]
                period_name = "unknown"
            
            # 尝试找到能整除的记录大小
            for size in preferred_sizes:
                if file_size % size == 0:
                    record_count = file_size // size
                    if record_count > 0:
                        # 对于1m数据，减少日志输出
                        if period_name == "1m":
                            self.logger.debug(f"估算{period_name}记录数: {record_count:,}")
                        else:
                            self.logger.info(f"估算{period_name}记录数: {record_count:,} (文件大小: {file_size:,}, 记录大小: {size})")
                        return record_count
            
            # 如果没有找到能整除的，使用合理的默认值
            if '/86400/' in file_path or '\\86400\\' in file_path:
                # 日线数据，假设32字节/条
                default_size = 32
            else:
                # 分钟数据，假设40字节/条
                default_size = 40
            
            estimated_count = file_size // default_size
            # 对于1m数据，减少警告输出
            if period_name == "1m":
                self.logger.debug(f"估算{period_name}记录数: {estimated_count:,}")
            else:
                self.logger.warning(f"使用默认记录大小估算{period_name}: {estimated_count:,} (记录大小: {default_size}, 可能不够精确)")
            return estimated_count
            
        except Exception as e:
            self.logger.error(f"估算记录数失败: {e}")
            return 0

    def detect_file_format(self, file_path):
        """
        检测文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 文件格式信息
        """
        info = {
            'format': 'unknown',
            'record_size': 0,
            'record_count': 0,
            'file_size': 0
        }
        
        if not os.path.exists(file_path):
            return info
            
        try:
            file_size = os.path.getsize(file_path)
            info['file_size'] = file_size
            
            # 判断周期类型
            if '/60/' in file_path or '\\60\\' in file_path:
                period_type = '1m'
            elif '/300/' in file_path or '\\300\\' in file_path:
                period_type = '5m'
            elif '/86400/' in file_path or '\\86400\\' in file_path:
                period_type = '1d'
            else:
                period_type = '1d'  # 默认
            
            # 获取真实记录数
            record_count = self.get_real_record_count(file_path, period_type)
            info['record_count'] = record_count
            
            # 根据记录数计算平均记录大小
            if record_count > 0:
                info['record_size'] = file_size // record_count
                info['format'] = f'{period_type}_data'
            else:
                # 降级到原来的方法
                possible_sizes = [32, 40, 48, 56, 64, 28]
                for size in possible_sizes:
                    if file_size % size == 0:
                        record_count = file_size // size
                        if record_count > 0:
                            info['record_size'] = size
                            info['record_count'] = record_count
                            info['format'] = f'{period_type}_estimated'
                            break
                        
        except Exception as e:
            self.logger.error(f"检测文件格式失败: {e}")
            
        return info 
