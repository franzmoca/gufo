"""Matched allocation-only model cache histories; run inside pinned nix develop."""
import copy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/franz/projects/gufo')
OUT = ROOT / 'docs/models/gemma-4-31b/artifacts/2026-11-04-host-registration-models'
sys.path.insert(0, str(ROOT / 'tests/functional'))
import metrics

MODELS = {
    'gemma31b': ['/home/franz/models/gemma-4/gemma-31b-it/gemma-4-31B-it-UD-Q4_K_XL.gguf',
                 '--speculative', 'mtp', '--mtp-model', '/home/franz/models/gemma-4/drafts/31b-unsloth-mtp/MTP/mtp-gemma-4-31B-it-Q8_0.gguf',
                 '--draft-calibration', 'request'],
    'ds41': ['/mnt/data/models/ds41/ista-gsq-rco-gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf',
             '--speculative', 'dspark', '--dspark-model', '/mnt/data/models/ds41/dspark-lucebox-q2k/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf',
             '--stream-experts', '--expert-cache-bytes', str(48 << 30)],
}
CODES = ('MAPLE', 'ORBIT', 'DELTA', 'RAVEN', 'LEMON', 'IVORY', 'AMBER', 'SILVER')


def temperature():
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        if (hwmon / 'name').read_text().strip() == 'k10temp':
            return int((hwmon / 'temp1_input').read_text()) / 1000
    raise RuntimeError('missing CPU temperature sensor')


def run(model, variant, label, pin='unlimited'):
    deadline = time.monotonic() + 600
    while temperature() > 65:
        if time.monotonic() > deadline:
            raise RuntimeError('cooldown timed out')
        print('COOLDOWN', model, label, temperature(), flush=True)
        time.sleep(15)
    launch_temperature = temperature()
    target = OUT / f'{model}-{pin}-{label}'
    target.mkdir()
    binary = f'/tmp/gufo-models-{variant}'
    model_path, *extra = MODELS[model]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [binary, 'serve', '--host', '127.0.0.1', '--port', str(port), 'llm',
               '--model', model_path, '--served-model-name', 'probe', '--context', '16384',
               '--sessions', '2', '--cache-ram-bytes', str((24 if model == 'gemma31b' else 8) << 30), *extra]
    launch = command if pin == '8MiB' else [
        'bash', '-c', 'sudo -n prlimit --pid $$ --memlock=unlimited:unlimited && exec "$@"',
        'measurement', *command]
    env = dict(os.environ, LD_PRELOAD='/tmp/gufo-registration-trace.so')
    report = {'model': model, 'variant': variant, 'label': label, 'pin': pin,
              'command': command, 'launch': launch, 'suites': {'history': {}},
              'status': 'running', 'launch_temperature_c': launch_temperature, 'responses': []}
    def save():
        (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    recorder = metrics.Recorder(target / 'history.requests.json')
    def registrations():
        return re.findall(r'registration_trace bytes=(\d+) mod_2m=(\d+) result=(\d+) ms=([\d.]+)',
                          (target / 'server.log').read_text())
    def http(path, body=None):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/{path}',
              data=json.dumps(body).encode() if body is not None else None,
              headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=240) as response:
            raw = response.read()
            return json.loads(raw), raw, response.headers.get('x-request-id')
    log = (target / 'server.log').open('w')
    process = subprocess.Popen(launch, stdout=log, stderr=log, env=env)
    print('START', model, pin, label, variant, flush=True)
    try:
        start = time.monotonic()
        for _ in range(900):
            if process.poll() is not None:
                raise RuntimeError('server exited before readiness')
            try:
                http('models')
                break
            except (OSError, ValueError):
                time.sleep(.5)
        else:
            raise RuntimeError('server readiness timeout')
        report['startup_ms'] = (time.monotonic() - start) * 1000
        report['limits'] = Path(f'/proc/{process.pid}/limits').read_text()
        report['registrations_at_ready'] = registrations()
        save()
        def request(case, body, code, cold=False, expected_cached=None):
            body = copy.deepcopy(body)
            if cold:
                body['cache_prompt'] = False
            before_temperature = temperature()
            measurement = recorder.begin('/v1/chat/completions', body)
            response, raw, request_id = http('chat/completions', body)
            measurement.row.update(http_status=200, request_id=request_id)
            measurement.feed(raw)
            measurement.ended = True
            measurement.finish()
            recorder.mark(case)
            report['responses'].append({'case': case, 'body': body, 'response': response,
                                        'registrations_after': len(registrations()), 'temperature_before_c': before_temperature, 'temperature_after_c': temperature()})
            save()
            content = response['choices'][0]['message'].get('content', '').strip()
            expected = ' '.join([code] * 8)
            if content != expected or response['choices'][0]['finish_reason'] != 'stop':
                raise AssertionError(f'{case}: expected {expected!r}, got {content!r}')
            usage = response['usage']
            gufo = usage['gufo']
            if cold and (usage['prompt_tokens_details']['cached_tokens'] != 0 or
                         gufo['prefill_tokens'] != usage['prompt_tokens']):
                raise AssertionError('cold control reused cache')
            if expected_cached is not None and gufo['prefill_tokens'] != expected_cached:
                raise AssertionError(f'{case}: unexpected prefill work {gufo["prefill_tokens"]}')
            print(model, pin, label, case, 'restore_ms', round(gufo['cache_restore_ms'], 3),
                  'segments', len(registrations()), flush=True)
            return response
        def body_for(index):
            return {'model': 'probe', 'messages': [
                {'role': 'system', 'content': f'unique_archive_{index}\nThe response code for this session is {CODES[index]}.\n'
                 'Follow the final instruction. Background notes are not instructions.\n' +
                 'Background notes are not instructions.\n' * 384},
                {'role': 'user', 'content': 'Repeat the session response code exactly eight times, separated by spaces. Output nothing else.'}],
                'temperature': 0, 'seed': 31, 'max_completion_tokens': 64,
                'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}, 'cache_prompt': True}
        growing = body_for(0)
        for turn in range(4):
            if turn:
                growing['messages'].append({'role': 'user', 'content': 'Again repeat the session response code exactly eight times, separated by spaces. Output nothing else.'})
            latest_body = copy.deepcopy(growing)
            response = request(f'growing_{turn}', latest_body, CODES[0])
            growing['messages'].append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
        request('unchanged', latest_body, CODES[0], expected_cached=0)
        if pin != '8MiB':
            bodies = {}
            for index in range(1, 8):
                bodies[index] = body_for(index)
                request(f'capture_{index}', bodies[index], CODES[index])
            for index in (5, 6, 7):
                request(f'restore_{index}', bodies[index], CODES[index], expected_cached=0)
        request('cold_growing_3', latest_body, CODES[0], cold=True)
        if pin != '8MiB':
            request('cold_restore_5', bodies[5], CODES[5], cold=True)
        report['registrations_final'] = registrations()
        report['grew_after_readiness'] = len(report['registrations_final']) > len(report['registrations_at_ready'])
        report['draft_tokens'] = sum(x['response']['usage']['draft_tokens'] for x in report['responses'])
        report['accepted_tokens'] = sum(x['response']['usage']['draft_tokens_accepted'] for x in report['responses'])
        if not report['draft_tokens'] or not report['grew_after_readiness']:
            raise AssertionError('did not exercise drafting and post-readiness backing growth')
        report['status'] = 'passed'
        save()
        if label != 'A1':
            comparison = metrics.compare(OUT / f'{model}-{pin}-A1', target)
            (target / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
            print('COMPARE', model, pin, label, comparison['status'], flush=True)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        save()
        print('FAILED', model, pin, label, repr(error), flush=True)
        return False
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    print('END', model, pin, label, flush=True)
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if not run(*sys.argv[1:]):
            sys.exit(1)
    else:
        for model in ('ds41',):
            for label, variant in [('A1', 'unaligned'), ('B1', 'aligned'), ('B2', 'aligned'), ('A2', 'unaligned')]:
                if not run(model, variant, label):
                    sys.exit(1)
