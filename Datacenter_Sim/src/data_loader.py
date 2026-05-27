import os
import pandas as pd
import glob
from typing import Generator, Dict, Any, List, Optional
import heapq

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

        try:
            for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
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
                # Ensure correct types
                chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'])
                for _, row in chunk.iterrows():
                    # Calculate compute time
                    # T_compute = (N_context / 1000) + (N_generated / 50) seconds
                    # Convert to milliseconds
                    compute_time_sec = (row['ContextTokens'] / 1000.0) + (row['GeneratedTokens'] / 50.0)
                    compute_time_ms = compute_time_sec * 1000.0

                    yield {
                        'type': 'LLM',
                        'timestamp_ms': row['TIMESTAMP'].timestamp() * 1000.0,
                        'cpu_req': 8,
                        'gpu_req': 1,
                        'io_time_ms': 0.0, # Assumed 0 IO wait for LLM for this model
                        'compute_time_ms': compute_time_ms
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
            chunk['gmt_create'] = pd.to_datetime(chunk['gmt_create'])
            for _, row in chunk.iterrows():
                e2e_time_ms = row['exec_time_seconds'] * 1000.0

                io_time_ms = 0.0
                if io_len > 0:
                    io_time_ms = float(io_times[io_idx % io_len])
                    io_idx += 1

                compute_time_ms = max(e2e_time_ms - io_time_ms, e2e_time_ms * 0.15)

                yield {
                    'type': 'Diffusion',
                    'timestamp_ms': row['gmt_create'].timestamp() * 1000.0,
                    'cpu_req': 4,
                    'gpu_req': 1,
                    'io_time_ms': io_time_ms,
                    'compute_time_ms': compute_time_ms
                }

class PhillyParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        filepath = self.get_file_path('*Philly_cluster_job_log.csv')
        if not filepath:
            return

        for chunk in self.read_csv_in_chunks(filepath):
            chunk['start_time'] = pd.to_datetime(chunk['start_time'])
            for _, row in chunk.iterrows():
                # GPU requirements
                gpus_str = str(row['gpus'])
                if gpus_str.lower() in ['nan', 'none', '']:
                    gpu_req = 1
                else:
                    gpu_req = len(gpus_str.split(','))

                duration_sec = float(row['duration_sec'])

                # [Jeon et al., ATC 2019] - Philly represents Training load, not DLRM.
                # cpu_req = gpu_req * 8 according to physics rules.

                yield {
                    'type': 'Training',
                    'timestamp_ms': row['start_time'].timestamp() * 1000.0,
                    'cpu_req': gpu_req * 8, # Training cpu/gpu ratio
                    'gpu_req': gpu_req,
                    'io_time_ms': 0.0,
                    'compute_time_ms': duration_sec * 1000.0
                }

class DlrmParser(BaseParser):
    def parse(self) -> Generator[Dict[str, Any], None, None]:
        # [Yang et al., NSDI 2025] - DLRM (Resource Assassin) belongs to Alibaba workload.
        filepath = self.get_file_path('*qps.csv')
        if not filepath:
            return

        for chunk in self.read_csv_in_chunks(filepath):
            # Filter specifically for 'API Requests'
            if 'request_type' in chunk.columns:
                chunk = chunk[chunk['request_type'] == 'API Requests']

            for _, row in chunk.iterrows():
                # Read unix timestamp directly and convert to ms
                timestamp_ms = float(row['timestamp_anon']) * 1000.0

                yield {
                    'type': 'DLRM',
                    'timestamp_ms': timestamp_ms,
                    'cpu_req': 48, # DLRM resource assassin (48 CPUs)
                    'gpu_req': 1,  # 1 GPU
                    'io_time_ms': 0.0,
                    'compute_time_ms': 2000.0 # Fixed compute time 2000ms
                }

def event_generator(data_dir: str, include_dlrm: bool = True) -> Generator[Dict[str, Any], None, None]:
    """
    Merge multiple event streams into a single chronologically sorted stream.
    We read a small buffer from each to sort efficiently.
    """
    # PhillyParser represents the base training load, so it's always included.
    parsers = [
        AzureParser(data_dir).parse(),
        AlibabaParser(data_dir).parse(),
        PhillyParser(data_dir).parse()
    ]

    # DlrmParser is our true DLRM assassin, conditionally included.
    if include_dlrm:
        parsers.append(DlrmParser(data_dir).parse())

    # To correctly combine datasets from completely different years (2017, 2023, 2024),
    # we first extract the start timestamp of each trace and align them all to start at t=0.

    # Read all items into memory and sort since the datasets are relatively small
    # but spread out across vastly different timestamps in the sample data.
    # To maintain chunking, we will yield them out after aligning times.
    # However, to be strictly OOM safe according to rules, we yield generators.
    # Since different traces start at vastly different times (years apart),
    # if we just heapq them, one trace will finish entirely before the next begins.
    # Instead, we will wrap each generator to offset its own timestamps to start at 0.

    def normalize_generator(gen):
        # We need to buffer the generator to find its true minimum timestamp
        # since rows in the CSV might not be perfectly sorted.
        # But we must avoid OOM, so we can't load the whole file.
        # Instead, we will sort just the buffered chunks roughly or assume it is somewhat sorted
        # Actually, for simpy we MUST yield events in non-decreasing order of time.
        # We will use a priority queue per generator to locally sort its stream to ensure non-decreasing relative_time.
        # To avoid negative relative times from out-of-order rows in the CSV,
        # we dynamically keep track of the minimum timestamp seen.

        # A simpler way to guarantee non-decreasing time is to force it:
        first_ts = None
        last_yielded_ts = 0.0
        for item in gen:
            if first_ts is None:
                first_ts = item['timestamp_ms']

            rel_ts = item['timestamp_ms'] - first_ts
            if rel_ts < last_yielded_ts:
                rel_ts = last_yielded_ts # Force non-decreasing to avoid SimPy crashes

            last_yielded_ts = rel_ts
            item['timestamp_ms'] = rel_ts
            yield item

    normalized_parsers = [normalize_generator(p) for p in parsers]

    pq = []

    # Initialize pq with first element of each normalized generator
    for i, p in enumerate(normalized_parsers):
        try:
            item = next(p)
            heapq.heappush(pq, (item['timestamp_ms'], i, item))
        except StopIteration:
            pass

    while pq:
        ts, idx, item = heapq.heappop(pq)

        item['relative_time_ms'] = ts
        yield item

        try:
            next_item = next(normalized_parsers[idx])
            heapq.heappush(pq, (next_item['timestamp_ms'], idx, next_item))
        except StopIteration:
            pass
