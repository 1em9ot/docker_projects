#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
my_parallel_executor.py

汎用的な並列処理モジュールです。
各タスクの実行結果と実行時間（elapsed）を返します。
ログ出力はオプションで有効化可能です。
"""

import concurrent.futures
import time
from datetime import datetime
import logging
from functools import wraps

def _timed_call(func, *args, **kwargs):
    """内部使用: 関数を実行し結果と実行時間をタプルで返す。"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return (result, elapsed)

def run_parallel_tasks(tasks, mode='thread', max_workers=None, profile=False, logger=None):
    """
    複数のタスクを並列実行する。
    各タスクは、callable または (callable, args, kwargs) の形式で指定する。
    プロファイリングが有効な場合、各タスクの実行時間を測定して結果とともに返す。
    """
    # ログ出力用関数の準備
    log = None
    file_obj = None
    if profile:
        if logger is None or logger == 'stdout':
            log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
        elif logger == 'stderr':
            import sys
            log = lambda msg: sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        elif isinstance(logger, str):
            file_obj = open(logger, 'a')
            log = lambda msg, f=file_obj: (f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"), f.flush())
        elif isinstance(logger, logging.Logger):
            log = lambda msg, lg=logger: lg.info(msg)
        elif hasattr(logger, "write"):
            log = lambda msg, f=logger: (f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"), f.flush())
        elif callable(logger):
            log = lambda msg, func=logger: func(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
        else:
            log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    # Executor選択
    ExecutorClass = concurrent.futures.ThreadPoolExecutor if mode == 'thread' else concurrent.futures.ProcessPoolExecutor

    results = []
    times = []  # 各タスクの実行時間を記録
    with ExecutorClass(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            if isinstance(task, (tuple, list)):
                func = task[0]
                args = task[1] if len(task) > 1 else ()
                kwargs = task[2] if len(task) > 2 else {}
                if profile:
                    futures.append(executor.submit(_timed_call, func, *args, **kwargs))
                else:
                    futures.append(executor.submit(func, *args, **kwargs))
            else:
                func = task
                if profile:
                    futures.append(executor.submit(_timed_call, func))
                else:
                    futures.append(executor.submit(func))
        for i, future in enumerate(futures):
            try:
                res = future.result()
            except Exception as e:
                if profile and log:
                    log(f"Task {i} raised an exception: {e}")
                results.append(e)
                if profile:
                    times.append(None)
            else:
                if profile:
                    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], (int, float)):
                        result_val, elapsed = res
                        results.append((result_val, elapsed))
                        times.append(elapsed)
                        if log:
                            log(f"Task {i} completed in {elapsed:.6f} seconds.")
                    else:
                        results.append((res, None))
                        times.append(None)
                        if log:
                            log(f"Task {i} completed.")
                else:
                    results.append(res)
    if profile and times:
        valid_times = [t for t in times if isinstance(t, (int, float))]
        if valid_times and log:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            log(f"Task execution times: min={min_time:.6f}s, max={max_time:.6f}s, avg={avg_time:.6f}s (n={len(valid_times)})")
    if file_obj:
        file_obj.close()
    return results

def profile_task(func=None, logger=None):
    """
    関数の実行時間をログ出力するデコレータ。
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            message = f"Function '{f.__name__}' executed in {elapsed:.6f} seconds."
            if logger is None:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            else:
                if isinstance(logger, logging.Logger):
                    logger.info(message)
                elif hasattr(logger, "write"):
                    logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
                    try:
                        logger.flush()
                    except:
                        pass
                elif callable(logger):
                    logger(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            return result
        return wrapper
    if func:
        return decorator(func)
    return decorator
