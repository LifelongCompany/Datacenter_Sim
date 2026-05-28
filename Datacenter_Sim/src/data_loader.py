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
            # 【修复1】添加 on_bad_lines='skip' 和 engine='c'，彻底无视断层脏数据！
            for chunk in pd.read_csv(filepath, chunksize=self.chunksize, on_bad_lines='skip', engine='c'):
                yield chunk
        except Exception as e:
            print(f"  [Error] 读取 {filepath} 时发生错误: {e}")


class AzureParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        # 【核心修复1】补全多模态数据集（注意微软的拼写是 LMM 不是 LLM）
        patterns = [
            '*AzureLLMInferenceTrace_code.csv',
            '*AzureLLMInferenceTrace_conv.csv',
            '*AzureLMMInferenceTrace_multimodal.csv'
        ]

        for pattern in patterns:
            filepath = self.get_file_path(pattern)
            if not filepath:
                continue

            for chunk in self.read_csv_in_chunks(filepath):
                # 【极速解析】开启 cache，遇到乱码行直接跳过，然后删掉无效行
                chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'], format='mixed', cache=True, errors='coerce')
                chunk = chunk.dropna(subset=['TIMESTAMP'])
                chunk = chunk.sort_values(by='TIMESTAMP')

                chunk['timestamp_ms'] = chunk['TIMESTAMP'].astype('int64') // 10 ** 6

                # 【核心护盾】防御空值引发的算术错误，防止静默宕机
                if 'ContextTokens' in chunk.columns:
                    chunk['ContextTokens'] = pd.to_numeric(chunk['ContextTokens'], errors='coerce').fillna(0)
                if 'GeneratedTokens' in chunk.columns:
                    chunk['GeneratedTokens'] = pd.to_numeric(chunk['GeneratedTokens'], errors='coerce').fillna(0)

                # 如果没有这些列，赋默认值
                ctx_tokens = chunk['ContextTokens'] if 'ContextTokens' in chunk.columns else 0
                gen_tokens = chunk['GeneratedTokens'] if 'GeneratedTokens' in chunk.columns else 0

                chunk['compute_time_ms'] = ((ctx_tokens / 1000.0) + (gen_tokens / 50.0)) * 1000.0

                # 【性能优化】极速元组遍历
                for row in chunk.itertuples(index=False):
                    yield {
                        'type': 'LLM',
                        'timestamp_ms': row.timestamp_ms,
                        'cpu_req': 8,
                        'gpu_req': 1,
                        'io_time_ms': 0.0,
                        'compute_time_ms': max(10.0, row.compute_time_ms)  # 保底 10ms 运算
                    }


class AlibabaParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        req_filepath = self.get_file_path('*lora_request_trace.csv')
        io_filepath = self.get_file_path('*basemodel_update_latency_anon.csv')

        io_times = []
        if io_filepath:
            try:
                for chunk in self.read_csv_in_chunks(io_filepath):
                    io_times.extend(chunk['value'].dropna().tolist())
            except Exception as e:
                print(f"  [Error] 加载 Alibaba IO 数据异常: {e}")

        io_idx = 0
        io_len = len(io_times)

        if not req_filepath:
            return

        for chunk in self.read_csv_in_chunks(req_filepath):
            chunk['gmt_create'] = pd.to_datetime(chunk['gmt_create'], format='mixed', cache=True, errors='coerce')
            chunk = chunk.dropna(subset=['gmt_create'])
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


class PhillyParser(BaseParser):
    def parse(self):
        filepath = self.get_file_path('*Philly_cluster_job_log*')
        if not filepath:
            return

        print(f"  [Loader] 弃用 ijson，开启底层极速 C-Decoder 流式解析: {os.path.basename(filepath)} ...")

        parsed_count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                char = f.read(1)
                while char and char != '[':
                    char = f.read(1)
                if not char:
                    print("  [Error] 未找到 JSON 数组的起始符 '['")
                    return

                buffer = ""
                decoder = json.JSONDecoder()

                while True:
                    chunk = f.read(1024 * 512)
                    if not chunk and not buffer.strip():
                        break
                    buffer += chunk

                    while True:
                        buffer = buffer.lstrip(', \n\r\t')
                        if not buffer:
                            break
                        if buffer.startswith(']'):
                            return

                        try:
                            job, idx = decoder.raw_decode(buffer)
                            buffer = buffer[idx:]
                        except json.JSONDecodeError:
                            break

                        try:
                            parsed_count += 1
                            if parsed_count == 1:
                                print("  [Loader] [FAST] 首条数据已秒速解开！引擎全速推进中...")

                            attempts = job.get('attempts', [])
                            if not attempts:
                                continue

                            attempt = attempts[0]
                            start_str = str(attempt.get('start_time', ''))
                            end_str = str(attempt.get('end_time', ''))

                            if not start_str or not end_str or start_str.lower() in ['none', 'null', 'nan',
                                                                                     ''] or end_str.lower() in ['none',
                                                                                                                'null',
                                                                                                                'nan',
                                                                                                                '']:
                                continue

                            # 【核心修复2】拆除毫秒级时间炸弹！兼容 "2017-10-03 14:00:00.123" 这种带小数的异常格式
                            start_str = start_str.split('.')[0].replace('T', ' ').strip()
                            end_str = end_str.split('.')[0].replace('T', ' ').strip()

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
                                'compute_time_ms': max(100.0, duration_sec * 1000.0)  # 防止计算时间为0引发无穷尽分配
                            }
                        except Exception as inner_e:
                            continue

                    if not chunk:
                        break

        except Exception as e:
            print(f"  [Error] reading Philly JSON: {e}")


class DlrmParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        filepath = self.get_file_path('*qps.csv')
        if not filepath:
            return
        for chunk in self.read_csv_in_chunks(filepath):
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
            aligned_ts = 0.0
            heapq.heappush(pq, (aligned_ts, i, item))
        except StopIteration:
            continue

    print(f"  [EventGen] 成功锚定 {len(pq)} 个数据流，开始对齐任务时间轴...")

    processed_count = 0
    while pq:
        aligned_ts, idx, item = heapq.heappop(pq)

        if processed_count == 0:
            print(f"  [EventGen] [FAST] 第一条数据已成功进入归并逻辑！Idx={idx}, Time={aligned_ts}", flush=True)

        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"  [EventGen] 进度反馈... 已归并: {processed_count} 个任务", flush=True)

        item['relative_time_ms'] = aligned_ts
        yield item

        try:
            next_item = next(parsers[idx])
            raw_next_ts = next_item['timestamp_ms'] - stream_offsets[idx]

            # 【修复点】探测到时间倒流（如切换到了新文件、或者 CSV 内部大范围乱序）
            if raw_next_ts < aligned_ts:
                # 动态自适应：重新校准该流的偏移锚点，让新文件的时间紧挨着当前时间自然推进
                stream_offsets[idx] = next_item['timestamp_ms'] - aligned_ts
                next_aligned_ts = aligned_ts
            else:
                next_aligned_ts = raw_next_ts

            heapq.heappush(pq, (next_aligned_ts, idx, next_item))
        except StopIteration:
            print(f"  [EventGen] 数据流 {idx} 已耗尽。")
            continue
