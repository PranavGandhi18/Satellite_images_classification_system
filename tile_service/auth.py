"""Optional access key for exposing the service through a shareable link (off unless Settings.access_key is set).

A request is let through if any of these carries the key:
  ?key=<key>                  (a shared link; on a GET the key is then moved into a cookie and stripped from the URL)
  X-API-Key: <key>            (scripts; does not clash with a reverse proxy that uses Authorization for itself)
  Authorization: Bearer <key>
  cookie tile_access=<key>    (set by the ?key redirect, so the browser page's own requests work)
Anything else gets 401: a small key-entry page for browsers, JSON for everything else.
"""
import hmac
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = 'tile_access'
COOKIE_MAX_AGE = 30 * 24 * 3600

LOCKED_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tile Classifier - access key required</title>
<style>body{font:15px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:90vh;margin:0 16px;background:#f6f7f9;color:#1d2330}
form{background:#fff;border:1px solid #e2e5eb;border-radius:10px;padding:24px;max-width:380px;width:100%}
input{width:100%;box-sizing:border-box;padding:8px;margin:8px 0 12px;font:inherit}button{padding:8px 16px;font:inherit}
@media(prefers-color-scheme:dark){body{background:#14171d;color:#e6e9ef}form{background:#1d212a;border-color:#2d333f}}</style></head>
<body><form method="get"><h2 style="margin-top:0">Access key required</h2>
<p>This link needs the key it was shared with. Open the full link, or paste the key here.</p>
<input name="key" type="password" autocomplete="off" placeholder="access key" required autofocus><button type="submit">Open</button></form></body></html>"""


def _matches(candidate: str | None, key: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), key.encode())


def install(app, key: str) -> None:
    @app.middleware('http')
    async def require_access_key(request: Request, call_next):
        from_query = request.query_params.get('key')
        auth = request.headers.get('authorization', '')
        bearer = auth[7:].strip() if auth.lower().startswith('bearer ') else None
        candidates = (from_query, request.headers.get('x-api-key'), bearer, request.cookies.get(COOKIE))
        if not any(_matches(c, key) for c in candidates):
            if request.method == 'GET' and 'text/html' in request.headers.get('accept', ''):
                return HTMLResponse(LOCKED_PAGE, status_code=401)
            return JSONResponse({'detail': 'access key required: open the shared link (?key=...) or send an X-API-Key header'},
                                status_code=401, headers={'WWW-Authenticate': 'Bearer'})
        if request.method == 'GET' and _matches(from_query, key):
            # move the key out of the URL (address bar, history, referrers, access logs) into an HttpOnly cookie
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != 'key']
            resp = RedirectResponse(request.url.path + ('?' + urlencode(rest) if rest else ''), status_code=303)
            resp.set_cookie(COOKIE, key, max_age=COOKIE_MAX_AGE, httponly=True, samesite='lax',
                            secure=request.url.scheme == 'https')
            return resp
        return await call_next(request)
