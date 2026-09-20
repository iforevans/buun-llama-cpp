#!/usr/bin/env python3
"""Bonsai server smoke checks, or fixed-length PP/TG probes without prompt reuse.

Each invocation owns one server process. Run reference and candidate serially on
an otherwise idle GPU. --perf-corpus takes a local text file (e.g. WikiText-2).
The performance probes are not a model-quality evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def run(args):
    binary = Path(args.binary).resolve(strict=True)
    with socket.socket() as check:
        if check.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError(f'port {args.port} is already occupied; choose another --port')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    command = [str(binary), '-m', args.model, '-ngl', str(args.gpu_layers),
               '-c', '8192', '-np', '1', '-b', '512', '-ub', '512', '-fa', 'on',
               '-ctk', args.kv, '-ctv', args.kv, '-t', '8',
               '--host', '127.0.0.1', '--port', str(args.port)]
    for override in args.override_tensor:
        command.extend(['-ot', override])
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    env = os.environ.copy()
    old_library_path = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = str(binary.parent) + (os.pathsep + old_library_path if old_library_path else '')

    def api(route, data=None):
        request = urllib.request.Request(
            f'http://127.0.0.1:{args.port}/{route}',
            data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.load(response)

    with (out / 'server.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited with {process.returncode}; see {out}/server.log')
                try:
                    api('health')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.5)

            if args.perf_corpus:
                text = Path(args.perf_corpus).read_text()
                tokens = api('tokenize', {'content': text, 'add_special': True})['tokens']
                if len(tokens) < 10240:
                    raise ValueError('performance corpus needs at least 10240 tokens')
                prompts = [('warmup', tokens[:32], 32)] + [
                    (f'pp{size}-r{rep}', tokens[rep * 4096:rep * 4096 + size], 256)
                    for size in (512, 2048) for rep in range(3)
                ]
            else:
                prompts = [
                    ('paris', 'Question: What is the capital of France? Answer:', 48),
                    ('math', 'Question: What is 7 plus 9? Answer:', 48),
                    ('code', 'Write a Python function that returns the square of an integer.\n', 128),
                    ('science', 'Explain why the sky appears blue during the day.\n', 128),
                    ('long', ('A garden contains apple trees, pears and herbs. Rain waters the soil.\n' * 80)
                     + 'Summarize the garden in two sentences.\n', 128),
                ]
            for name, prompt, count in prompts:
                if not args.perf_corpus:
                    prompt = api('apply-template', {
                        'messages': [{'role': 'user', 'content': prompt}],
                        'add_generation_prompt': True,
                        'chat_template_kwargs': {'enable_thinking': False}})['prompt']
                started = time.monotonic()
                result = api('completion', {
                    'prompt': prompt, 'n_predict': count, 'temperature': 0, 'seed': 1234,
                    'cache_prompt': False, 'n_probs': 0 if args.perf_corpus else 5,
                    'return_tokens': True, 'ignore_eos': bool(args.perf_corpus)})
                result['wall_seconds'] = time.monotonic() - started
                (out / f'{name}.json').write_text(json.dumps(result, indent=2))
                if args.perf_corpus:
                    timing = result['timings']
                    if timing.get('cache_n', 0) != 0 or timing['prompt_n'] != len(prompt):
                        raise RuntimeError(f'{name}: prompt was not evaluated in full')
                    if timing['predicted_n'] != count:
                        raise RuntimeError(f'{name}: generation stopped before its fixed token budget')
                print(json.dumps({'name': name, 'timings': result.get('timings'),
                                  'text': result.get('content', '')[:300]}), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True, help='new artifact directory; never overwritten')
    parser.add_argument('--port', type=int, default=8130)
    parser.add_argument('--override-tensor', action='append', default=[])
    parser.add_argument('--kv', default='f16')
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--perf-corpus', help='enable fixed-length performance probes')
    run(parser.parse_args())
