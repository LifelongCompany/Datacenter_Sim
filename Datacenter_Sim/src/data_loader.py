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

                yield {
                    'type': 'DLRM',
                    'timestamp_ms': row['start_time'].timestamp() * 1000.0,
                    'cpu_req': 48, # DLRM is a resource assassin
                    'gpu_req': 1, # As per rules
                    'io_time_ms': 0.0,
                    'compute_time_ms': duration_sec * 1000.0
                }

def event_generator(data_dir: str, include_dlrm: bool = True) -> Generator[Dict[str, Any], None, None]:
    """
    Merge multiple event streams into a single chronologically sorted stream.
    We read a small buffer from each to sort efficiently.
    """
    parsers = [
        AzureParser(data_dir).parse(),
        AlibabaParser(data_dir).parse()
    ]
    if include_dlrm:
        parsers.append(PhillyParser(data_dir).parse())

    # We can use heapq to merge sorted iterators, but our generators are not
    # strictly sorted globally across chunks if we don't sort chunks.
    # To be perfectly correct with streaming, we'll buffer and sort.
    # Since data spans various days, we assume the dataset chunks are roughly sorted by time.

    pq = []

    # Initialize pq with first element of each generator
    for i, p in enumerate(parsers):
        try:
            item = next(p)
            heapq.heappush(pq, (item['timestamp_ms'], i, item))
        except StopIteration:
            pass

    first_timestamp = None

    while pq:
        ts, idx, item = heapq.heappop(pq)

        if first_timestamp is None:
            first_timestamp = ts

        # Convert timestamp to relative offset from the start of the simulation
        item['relative_time_ms'] = ts - first_timestamp
        yield item

        try:
            next_item = next(parsers[idx])
            heapq.heappush(pq, (next_item['timestamp_ms'], idx, next_item))
        except StopIteration:
            pass
