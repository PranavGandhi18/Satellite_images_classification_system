"""Network guard used by scripts/offline_check.py. NOT part of the service.

Python imports `sitecustomize` automatically at startup when this directory is on PYTHONPATH. It then blocks - and
logs to $NETGUARD_LOG - every DNS lookup, TCP connection and UDP send that is not to localhost. A process that works
normally under this guard, with nothing logged as blocked, did not try to use the network.
"""
import json
import os
import socket
import sys
import time

_LOG = os.environ.get('NETGUARD_LOG')
_LOCAL_NAMES = {'localhost', 'localhost.localdomain', 'ip6-localhost', 'ip6-loopback'}


def _log(event, target=None):
    if _LOG:
        with open(_LOG, 'a') as f:
            f.write(json.dumps({'t': round(time.time(), 3), 'pid': os.getpid(), 'event': event,
                                'target': repr(target), 'argv': sys.argv[:4]}) + '\n')


def _is_local(host) -> bool:
    if host is None or host == '' or host == b'':
        return True
    h = host.decode() if isinstance(host, bytes) else str(host)
    return h in _LOCAL_NAMES or h.startswith('127.') or h in ('::1', '0.0.0.0', '::')


def _check(sock, address, event):
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    if not _is_local(address[0]):
        _log('blocked_' + event, address)
        raise OSError(101, f'netguard: {event} to {address!r} blocked (offline mode)')
    _log('local_' + event, address)  # allowed, but recorded: e.g. an attempt via a local proxy would show up here


_connect, _connect_ex, _sendto = socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto
_getaddrinfo, _gethostbyname, _gethostbyname_ex = socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex


def connect(self, address):
    _check(self, address, 'connect')
    return _connect(self, address)


def connect_ex(self, address):
    _check(self, address, 'connect')
    return _connect_ex(self, address)


def sendto(self, data, *args):
    _check(self, args[-1], 'sendto')
    return _sendto(self, data, *args)


def _resolver(original, name):
    def resolve(host, *args, **kwargs):
        if not _is_local(host):
            _log('blocked_dns', host)
            raise socket.gaierror(socket.EAI_NONAME, f'netguard: DNS lookup of {host!r} blocked (offline mode)')
        return original(host, *args, **kwargs)
    resolve.__name__ = name
    return resolve


socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = connect, connect_ex, sendto
socket.getaddrinfo = _resolver(_getaddrinfo, 'getaddrinfo')
socket.gethostbyname = _resolver(_gethostbyname, 'gethostbyname')
socket.gethostbyname_ex = _resolver(_gethostbyname_ex, 'gethostbyname_ex')
_log('guard_loaded')
