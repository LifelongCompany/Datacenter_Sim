import os
import pandas as pd
import glob
from typing import Generator, Dict, Any, List, Optional
import heapq
import json
from datetime import datetime

class BaseParser:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunksize = 100000

    def get_file_path(self, pattern: str) -> Optional[str]:
        import glob
        search_path = os.path.join(self.data_dir, pattern)
        files = glob.glob(search_path)
        if files:
            return files[0]
        return None

    def read_csv_in_chunks(self, filepath: str) -> Generator[pd.DataFrame, None, None]:
        if filepath is None or not os.path.exists(filepath):
            return

        print(f"  [Loader] 预热加载中: {os.path.basename(filepath)} ...")
        try:
            # 【修复1】添加 on_bad_lines='skip' 和 engine='c'，彻底无视像第34行那样的断层脏数据！
            for chunk in pd.read_csv(filepath, chunksize=self.chunksize, on_bad_lines='skip', engine='c'):
                yield chunk
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


class AzureParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        patterns = ['*AzureLLMInferenceTrace_code.csv', '*AzureLLMInferenceTrace_conv.csv']
        for pattern in patterns:
            filepath = self.get_file_path(pattern)
            if not filepath:
                continue

            for chunk in self.read_csv_in_chunks(filepath):
                # 【极速解析】开启 cache，遇到乱码行直接跳过(coerce会转为NaT)，然后删掉无效行
                chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'], format='mixed', cache=True, errors='coerce')
                chunk = chunk.dropna(subset=['TIMESTAMP'])  # 丢弃坏数据

                chunk['timestamp_ms'] = chunk['TIMESTAMP'].astype('int64') // 10 ** 6
                chunk['compute_time_ms'] = ((chunk['ContextTokens'] / 1000.0) + (
                            chunk['GeneratedTokens'] / 50.0)) * 1000.0

                # 【性能优化】弃用 iterrows()，改用极速的 itertuples()
                for row in chunk.itertuples(index=False):
                    yield {
                        'type': 'LLM',
                        'timestamp_ms': row.timestamp_ms,
                        'cpu_req': 8,
                        'gpu_req': 1,
                        'io_time_ms': 0.0,
                        'compute_time_ms': row.compute_time_ms
                    }

class AlibabaParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        # Process requests and randomly assign an IO wait from basemodel if needed
        # In a real scenario, this might need matching, but let's sample or assume
        # For simplicity, we just yield requests.
        req_filepath = self.get_file_path('*lora_request_trace.csv')
        io_filepath = self.get_file_path('*basemodel_update_latency_anon.csv')

        # Load IO times safely with chunking to avoid OOM
        io_times = []
        if io_filepath:
            try:
                for chunk in self.read_csv_in_chunks(io_filepath):
                    io_times.extend(chunk['value'].dropna().tolist())
            except Exception as e:
                print(f"Error loading IO times safely: {e}")

        io_idx = 0
        io_len = len(io_times)

        if not req_filepath:
            return

        for chunk in self.read_csv_in_chunks(req_filepath):
            # 【极速解析】Alibaba 的 gmt_create
            chunk['gmt_create'] = pd.to_datetime(chunk['gmt_create'], format='mixed', cache=True, errors='coerce')
            chunk = chunk.dropna(subset=['gmt_create'])

            # 使用 itertuples 提速（如果你之前没改这个 parser 的话）
            chunk['timestamp_ms'] = chunk['gmt_create'].astype('int64') // 10 ** 6
            for row in chunk.itertuples(index=False):
                e2e_time_ms = float(row.exec_time_seconds) * 1000.0
                io_time_ms = 0.0
                if io_len > 0:
                    io_time_ms = float(io_times[io_idx % io_len])
                    io_idx += 1
                compute_time_ms = max(e2e_time_ms - io_time_ms, e2e_time_ms * 0.15)

                yield {
                    'type': 'Diffusion',
                    'timestamp_ms': row.timestamp_ms,
                    'cpu_req': 4,
                    'gpu_req': 1,
                    'io_time_ms': io_time_ms,
                    'compute_time_ms': compute_time_ms
                }


import json
from datetime import datetime
import os

import json
from datetime import datetime
import os


class PhillyParser(BaseParser):
    def parse(self):
        filepath = self.get_file_path('*Philly_cluster_job_log*')
        if not filepath:
            return

        print(f"  [Loader] 弃用 ijson，开启底层极速 C-Decoder 流式解析: {os.path.basename(filepath)} ...")

        parsed_count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # 1. 快速寻找数组开头 '['
                char = f.read(1)
                while char and char != '[':
                    char = f.read(1)
                if not char:
                    print("  [Error] 未找到 JSON 数组的起始符 '['")
                    return

                buffer = ""
                decoder = json.JSONDecoder()

                # 2. 建立 512KB 的极速滑动窗口
                while True:
                    chunk = f.read(1024 * 512)
                    if not chunk and not buffer.strip():
                        break
                    buffer += chunk

                    # 3. 疯狂提取完整的 JSON 对象
                    while True:
                        buffer = buffer.lstrip(', \n\r\t')
                        if not buffer:
                            break
                        if buffer.startswith(']'):
                            # 【核心修复2】完美识别 JSON 数组结束符，直接终结生成器，防止文件末尾死循环
                            return

                        try:
                            # 极速切除合法的 JSON 字典
                            job, idx = decoder.raw_decode(buffer)
                            buffer = buffer[idx:]
                        except json.JSONDecodeError:
                            break  # 缓冲区数据不完整，去读下一块

                        # === 【新增：内层防弹装甲】 ===
                        try:
                            parsed_count += 1
                            if parsed_count == 1:
                                print("  [Loader] ⚡ 首条数据已秒速解开！引擎全速推进中...")

                            attempts = job.get('attempts', [])
                            if not attempts:
                                continue

                            attempt = attempts[0]
                            # 强转为字符串，防止取到真实的 Python None
                            start_str = str(attempt.get('start_time', ''))
                            end_str = str(attempt.get('end_time', ''))

                            # 【核心修复】拦截 'None', 'null', 'NaN' 或空值
                            if not start_str or not end_str or start_str.lower() in ['none', 'null', 'nan',
                                                                                     ''] or end_str.lower() in ['none',
                                                                                                                'null',
                                                                                                                'nan',
                                                                                                                '']:
                                continue

                            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                            duration_sec = (end_dt - start_dt).total_seconds()

                            gpu_req = 0
                            for detail in attempt.get('detail', []):
                                gpu_req += len(detail.get('gpus', []))

                            if gpu_req == 0:
                                gpu_req = 1

                            yield {
                                'type': 'Training',
                                'timestamp_ms': start_dt.timestamp() * 1000.0,
                                'cpu_req': gpu_req * 8,
                                'gpu_req': gpu_req,
                                'io_time_ms': 0.0,
                                'compute_time_ms': max(0.0, duration_sec * 1000.0)
                            }
                        except Exception as inner_e:
                            # 【核心护盾】无论这条数据里的时间格式有多离谱，静默吞掉异常，继续解析下一条！
                            continue

                    if not chunk:
                        break

        except Exception as e:
            print(f"Error reading Philly JSON: {e}")

class DlrmParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        filepath = self.get_file_path('*qps.csv')
        if not filepath:
            return
        for chunk in self.read_csv_in_chunks(filepath):
            # 过滤刺客任务
            dlrm_chunk = chunk[chunk['request_type'] == 'API Requests']
            for row in dlrm_chunk.itertuples(index=False):
                yield {
                    'type': 'DLRM',
                    'timestamp_ms': float(row.timestamp_anon) * 1000.0,
                    'cpu_req': 48,
                    'gpu_req': 1,
                    'io_time_ms': 0.0,
                    'compute_time_ms': 2000.0
                }


def event_generator(data_dir: str, include_dlrm: bool = True) -> Generator[Dict[str, Any], None, None]:
    """
    带状态监控的多路归并时间轴生成器
    通过 heapq 维护一个最小堆，保证不同来源的任务流按时间顺序线性输出
    """
    parsers = [
        AzureParser(data_dir).parse(),
        AlibabaParser(data_dir).parse(),
        PhillyParser(data_dir).parse()
    ]
    if include_dlrm:
        try:
            parsers.append(DlrmParser(data_dir).parse())
        except NameError:
            pass

    pq = []
    stream_offsets = {}

    print("  [EventGen] 正在初始化多路时间轴对齐，请稍候...")

    # 1. 初始化阶段：每个数据源提取第一条记录作为初始锚点
    for i, p in enumerate(parsers):
        try:
            item = next(p)
            stream_offsets[i] = item['timestamp_ms']
            # 折叠：所有数据源的起始时间都归零
            aligned_ts = 0.0
            heapq.heappush(pq, (aligned_ts, i, item))
        except StopIteration:
            continue

    print(f"  [EventGen] 成功锚定 {len(pq)} 个数据流，开始对齐任务时间轴...")


    processed_count = 0
    # 2. 循环阶段
    # ... 在 event_generator 内部 ...
    while pq:
        aligned_ts, idx, item = heapq.heappop(pq)

        if processed_count == 0:
            # 强制刷新缓冲区！
            print(f"  [EventGen] ⚡ 第一条数据已成功进入归并逻辑！Idx={idx}, Time={aligned_ts}", flush=True)

        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"  [EventGen] 进度反馈... 已归并: {processed_count} 个任务", flush=True)

        item['relative_time_ms'] = aligned_ts
        yield item

        try:
            # 【核心排查点】：如果我们卡在这里，说明 next(parsers[idx]) 永远无法返回
            next_item = next(parsers[idx])
            next_aligned_ts = next_item['timestamp_ms'] - stream_offsets[idx]
            next_aligned_ts = max(aligned_ts, next_aligned_ts)
            heapq.heappush(pq, (next_aligned_ts, idx, next_item))
        except StopIteration:
            print(f"  [EventGen] 数据流 {idx} 已耗尽。")
            continue
        except Exception as e:
            print(f"  [EventGen] 数据流 {idx} 在读取下一条时发生异常: {e}")
            continue
